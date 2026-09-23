"""Reading, splitting and merging YOLO datasets.

The point of this module is combining your own annotations with a dataset from
somewhere else — a Roboflow export, say. The hard part is not copying files: it
is that two datasets number their classes independently, so ``0`` means "car" in
one and "person" in the other. Labels are rewritten against a unified class list
as they are merged.

Three directory layouts are understood, because they are the ones in the wild::

    roboflow:  <root>/train/images/*.jpg    <root>/train/labels/*.txt
               <root>/valid/…               <root>/test/…
    ours:      <root>/images/train/*.jpg    <root>/labels/train/*.txt
    flat:      <root>/images/*.jpg          <root>/labels/*.txt
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")

# Roboflow calls it "valid"; ultralytics and this tool call it "val".
SPLIT_ALIASES = {"valid": "val", "validation": "val", "training": "train", "testing": "test"}


@dataclass
class Item:
    """One image and the label file that goes with it."""

    image: Path
    label: Optional[Path]
    split: Optional[str]
    source: str

    @property
    def stem(self) -> str:
        return self.image.stem


@dataclass
class Dataset:
    root: Path
    names: List[str]
    items: List[Item]
    layout: str
    source: str = "dataset"
    warnings: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        counts = {split: 0 for split in SPLITS}
        counts["unassigned"] = 0
        for item in self.items:
            counts[item.split or "unassigned"] = counts.get(item.split or "unassigned", 0) + 1
        return {key: value for key, value in counts.items() if value}

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            for class_id, _ in read_label(item.label):
                name = self.names[class_id] if class_id < len(self.names) else f"class_{class_id}"
                counts[name] = counts.get(name, 0) + 1
        return counts


# --------------------------------------------------------------------- reading


def extract_zip(archive: Path, target: Path) -> Path:
    """Unpack a dataset zip, refusing entries that point outside the target."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if target.resolve() not in destination.parents and destination != target.resolve():
                raise ValueError(f"refusing zip entry outside the target: {member.filename}")
        bundle.extractall(target)
    return target


def _dataset_root(path: Path) -> Path:
    """Look through a single wrapper directory, which zips often add."""
    if _detect_layout(path) != "unknown":
        return path
    children = [child for child in path.iterdir() if child.is_dir()]
    visible = [child for child in children if not child.name.startswith((".", "__"))]
    if len(visible) == 1 and _detect_layout(visible[0]) != "unknown":
        return visible[0]
    return path


def _detect_layout(root: Path) -> str:
    if any((root / split / "images").is_dir() for split in ("train", "valid", "val", "test")):
        return "roboflow"
    if (root / "images").is_dir() and any(
        (root / "images" / split).is_dir() for split in ("train", "valid", "val", "test")
    ):
        return "split-last"
    if (root / "images").is_dir():
        return "flat"
    return "unknown"


def read_names(root: Path) -> Tuple[List[str], List[str]]:
    """Class names from data.yaml, falling back to classes.txt."""
    warnings: List[str] = []
    for candidate in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        names = _names_from_yaml(candidate)
        if names:
            return names, warnings
    classes_file = root / "classes.txt"
    if classes_file.exists():
        names = [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]
        if names:
            return names, warnings
    warnings.append(
        f"no data.yaml or classes.txt in {root}; classes will be named class_<n>"
    )
    return [], warnings


def _names_from_yaml(path: Path) -> Optional[List[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import yaml

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return _normalize_names(data.get("names"))
    except ImportError:
        pass
    return _names_from_yaml_text(text)


def _normalize_names(raw) -> Optional[List[str]]:
    if isinstance(raw, dict):
        try:
            return [str(raw[key]) for key in sorted(raw, key=lambda k: int(k))]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, (list, tuple)):
        return [str(name) for name in raw]
    return None


def _names_from_yaml_text(text: str) -> Optional[List[str]]:
    """Enough YAML for the two shapes data.yaml files actually use."""
    names: Dict[int, str] = {}
    collecting = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("names:"):
            inline = line.split(":", 1)[1].strip()
            if inline.startswith("["):          # names: ['cat', 'dog']
                inside = inline.strip("[]")
                parsed = [part.strip().strip("'\"") for part in inside.split(",")]
                return [name for name in parsed if name]
            collecting = True
            continue
        if collecting:
            if not line.startswith((" ", "\t", "-")):
                break
            if stripped.startswith("- "):        # names:\n  - cat\n  - dog
                names[len(names)] = stripped[2:].strip().strip("'\"")
            elif ":" in stripped:                # names:\n  0: cat
                key, _, value = stripped.partition(":")
                try:
                    names[int(key.strip())] = value.strip().strip("'\"")
                except ValueError:
                    continue
    return [names[key] for key in sorted(names)] if names else None


def label_for(image: Path, images_dir: Path, labels_dir: Path) -> Optional[Path]:
    candidate = labels_dir / (image.stem + ".txt")
    return candidate if candidate.exists() else None


def read_label(path: Optional[Path]) -> List[Tuple[int, str]]:
    """Parse a YOLO label file into (class id, rest of the line) pairs.

    The rest of the line is kept verbatim so segmentation polygons and
    oriented boxes survive a merge unchanged.
    """
    if path is None or not path.exists():
        return []
    rows: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        try:
            rows.append((int(float(head)), rest.strip()))
        except ValueError:
            continue  # not a YOLO row; skip rather than corrupt the merge
    return rows


def load_dataset(path, source: Optional[str] = None, extract_to: Optional[Path] = None) -> Dataset:
    """Read a YOLO dataset from a directory or a zip file."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such dataset: {path}")

    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ValueError(f"expected a directory or a .zip, got {path.name}")
        target = Path(extract_to or path.with_suffix(""))
        path = _dataset_root(extract_zip(path, target))
    else:
        path = _dataset_root(path)

    layout = _detect_layout(path)
    if layout == "unknown":
        raise ValueError(
            f"{path} does not look like a YOLO dataset (no images/ or train/images/ "
            f"directory). Roboflow's COCO and VOC exports are not YOLO — "
            f"re-download the zip in a YOLO format."
        )

    names, warnings = read_names(path)
    items: List[Item] = []
    tag = source or path.name

    if layout == "roboflow":
        for split_dir in sorted(path.iterdir()):
            if not split_dir.is_dir():
                continue
            split = SPLIT_ALIASES.get(split_dir.name.lower(), split_dir.name.lower())
            images_dir, labels_dir = split_dir / "images", split_dir / "labels"
            if not images_dir.is_dir():
                continue
            items += _items_in(images_dir, labels_dir, split if split in SPLITS else None, tag)
    elif layout == "split-last":
        for split_dir in sorted((path / "images").iterdir()):
            if not split_dir.is_dir():
                continue
            split = SPLIT_ALIASES.get(split_dir.name.lower(), split_dir.name.lower())
            labels_dir = path / "labels" / split_dir.name
            items += _items_in(images_dir=split_dir, labels_dir=labels_dir,
                               split=split if split in SPLITS else None, source=tag)
    else:
        items += _items_in(path / "images", path / "labels", None, tag)

    if not names:
        highest = max(
            (class_id for item in items for class_id, _ in read_label(item.label)),
            default=-1,
        )
        names = [f"class_{index}" for index in range(highest + 1)]

    missing = sum(1 for item in items if item.label is None)
    if missing:
        warnings.append(f"{missing} image(s) in {tag} have no label file (treated as empty)")
    return Dataset(root=path, names=names, items=items, layout=layout,
                   source=tag, warnings=warnings)


def _items_in(images_dir: Path, labels_dir: Path, split: Optional[str], source: str) -> List[Item]:
    if not images_dir.is_dir():
        return []
    items = []
    for image in sorted(images_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            items.append(Item(image=image, label=label_for(image, images_dir, labels_dir),
                              split=split, source=source))
    return items


# -------------------------------------------------------------------- splitting


def assign_split(key: str, ratios: Sequence[float]) -> str:
    """Deterministically place one item into train/val/test.

    Hashing the name (rather than shuffling) means re-running a split after
    adding images leaves every existing image where it was.
    """
    train, val, test = ratios
    total = train + val + test
    if total <= 0:
        return "train"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    position = (int.from_bytes(digest[:6], "big") % 1_000_000) / 1_000_000 * total
    if position < val:
        return "val"
    if position < val + test:
        return "test"
    return "train"


def normalize_ratios(train: Optional[float], val: float, test: float) -> Tuple[float, float, float]:
    if val < 0 or test < 0:
        raise ValueError("split fractions cannot be negative")
    if val + test >= 1.0:
        raise ValueError(f"val + test must leave room for training data (got {val + test})")
    return (1.0 - val - test if train is None else train, val, test)


# ---------------------------------------------------------------------- merging


@dataclass
class MergeReport:
    out_dir: Path
    names: List[str]
    per_source: Dict[str, Dict[str, int]]
    per_split: Dict[str, int]
    renamed: int
    class_origin: Dict[str, List[str]]
    warnings: List[str]
    duplicates: List[str] = field(default_factory=list)
    dropped: int = 0

    def describe(self) -> str:
        lines = [f"merged dataset written to {self.out_dir}", ""]
        for source, counts in self.per_source.items():
            detail = ", ".join(f"{split}: {count}" for split, count in counts.items())
            lines.append(f"  {source:<24} {detail}")
        lines.append("")
        lines.append("  splits   " + ", ".join(f"{k}: {v}" for k, v in self.per_split.items()))
        lines.append(f"  classes  {len(self.names)}")
        for index, name in enumerate(self.names):
            origin = ", ".join(self.class_origin.get(name, []))
            lines.append(f"    {index:>3}  {name:<24} from {origin}")
        if self.renamed:
            lines.append(f"\n  {self.renamed} file(s) renamed to avoid collisions")
        if self.dropped:
            lines.append(f"  {self.dropped} duplicate image(s) dropped")
        elif self.duplicates:
            lines.append(
                f"\n  ! {len(self.duplicates)} image(s) are byte-identical to another "
                f"image in the merge — the same picture in train and val inflates "
                f"your scores. Re-run with --drop-duplicates to keep only the first."
            )
            for line in self.duplicates[:5]:
                lines.append(f"      {line}")
            if len(self.duplicates) > 5:
                lines.append(f"      … and {len(self.duplicates) - 5} more")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


def unify_names(datasets: Sequence[Dataset], class_map: Optional[Dict[str, str]] = None):
    """Build one class list across datasets, keeping the first one's order.

    ``class_map`` renames classes on the way in, which is how two datasets that
    call the same thing "Car" and "car" end up as one class.
    """
    class_map = {key.strip(): value.strip() for key, value in (class_map or {}).items()}
    names: List[str] = []
    origin: Dict[str, List[str]] = {}
    remaps: List[Dict[int, int]] = []

    for dataset in datasets:
        remap: Dict[int, int] = {}
        for index, raw in enumerate(dataset.names):
            name = class_map.get(raw, raw)
            if name not in names:
                names.append(name)
            remap[index] = names.index(name)
            origin.setdefault(name, [])
            if dataset.source not in origin[name]:
                origin[name].append(dataset.source)
        remaps.append(remap)
    return names, remaps, origin


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_datasets(
    sources: Sequence[Dataset],
    out_dir: Path,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    resplit: bool = False,
    image_mode: str = "copy",
    class_map: Optional[Dict[str, str]] = None,
    drop_duplicates: bool = False,
) -> MergeReport:
    """Combine datasets into one, rewriting class ids onto a shared list.

    Items that already carry a split keep it unless ``resplit`` is set — a
    downloaded dataset's validation images are held out for a reason, and
    reshuffling them against your own images leaks them into training.
    """
    if image_mode not in ("copy", "link"):
        raise ValueError(f"unknown image mode {image_mode!r}; use copy or link")

    out_dir = Path(out_dir)
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    names, remaps, origin = unify_names(sources, class_map)
    per_source: Dict[str, Dict[str, int]] = {}
    per_split: Dict[str, int] = {split: 0 for split in SPLITS}
    warnings: List[str] = [w for dataset in sources for w in dataset.warnings]
    taken: Dict[str, str] = {}
    renamed = 0
    # Byte-identical images across sources are the real hazard of merging: the
    # same picture in train and val quietly inflates every metric.
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    dropped = 0

    for dataset, remap in zip(sources, remaps):
        counts: Dict[str, int] = {}
        for item in dataset.items:
            digest = _content_hash(item.image)
            if digest in seen:
                duplicates.append(f"{dataset.source}/{item.image.name} == {seen[digest]}")
                if drop_duplicates:
                    dropped += 1
                    continue
            else:
                seen[digest] = f"{dataset.source}/{item.image.name}"

            split = (
                assign_split(f"{dataset.source}/{item.stem}", ratios)
                if resplit or item.split is None
                else item.split
            )
            stem = item.stem
            if stem in taken and taken[stem] != dataset.source:
                stem = f"{_tag(dataset.source)}_{stem}"    # two datasets, one filename
                renamed += 1
            taken.setdefault(stem, dataset.source)

            target_image = out_dir / "images" / split / (stem + item.image.suffix)
            _place(item.image, target_image, image_mode)

            rows = [
                f"{remap.get(class_id, class_id)} {rest}".strip()
                for class_id, rest in read_label(item.label)
            ]
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )
            counts[split] = counts.get(split, 0) + 1
            per_split[split] = per_split.get(split, 0) + 1
        per_source[dataset.source] = counts

    write_data_yaml(out_dir, names, per_split)
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    if not per_split.get("train"):
        warnings.append(
            "the merged dataset has NO TRAINING IMAGES — nothing can be trained "
            "on it. Check the split fractions and any dropped duplicates."
        )
    if not per_split.get("val"):
        warnings.append("the merged dataset has no validation images")
    if dropped and dropped > (dropped + sum(per_split.values())) / 2:
        warnings.append(
            f"{dropped} of {dropped + sum(per_split.values())} images were dropped as "
            f"duplicates — that is most of the data, so check the sources overlap "
            f"as much as that implies"
        )
    return MergeReport(out_dir=out_dir, names=names, per_source=per_source,
                       per_split={k: v for k, v in per_split.items() if v},
                       renamed=renamed, class_origin=origin, warnings=warnings,
                       duplicates=duplicates, dropped=dropped)


def _tag(source: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in source)[:24]


def _place(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "link":
        try:
            target.symlink_to(source.resolve())
            return
        except (OSError, NotImplementedError):
            pass  # Windows without developer mode, or across filesystems
    shutil.copy2(source, target)


def write_data_yaml(out_dir: Path, names: Sequence[str], per_split: Dict[str, int]) -> Path:
    """The dataset descriptor ultralytics reads."""
    out_dir = Path(out_dir)
    lines = ["# written by auto-annotator", f"path: {out_dir.resolve()}", "train: images/train"]
    lines.append(f"val: images/{'val' if per_split.get('val') else 'train'}")
    if per_split.get("test"):
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
