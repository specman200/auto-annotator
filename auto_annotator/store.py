"""On-disk project: an image folder plus a sidecar directory of annotations.

Layout::

    <images>/
        cat.jpg
        nested/dog.png
        .auto-annotator/
            project.json          # classes + settings
            annotations/
                cat.json
                nested__dog.json

Annotations live in their own files so a crash, or two people working on
different images, can never take out the whole project.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import (
    STATUS_NEW,
    STATUS_PREDICTED,
    STATUS_REVIEWED,
    Annotation,
    ImageRecord,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_DIRNAME = ".auto-annotator"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _slug(relpath: str) -> str:
    """Flatten a relative image path into a single sidecar filename."""
    return relpath.replace("/", "__").replace("\\", "__")


def image_size(path: Path) -> tuple:
    """Return (width, height), or (0, 0) if the file is not a readable image."""
    try:
        from PIL import Image  # imported lazily so schema-only use needs no Pillow

        with Image.open(path) as img:
            return img.size
    except Exception:
        return (0, 0)


class Project:
    """All the state for one annotation session, backed by a directory."""

    def __init__(self, image_root: Path, classes: Optional[List[str]] = None):
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"not a directory: {self.image_root}")
        self.project_dir = self.image_root / PROJECT_DIRNAME
        self.annotation_dir = self.project_dir / "annotations"
        self._lock = threading.RLock()
        self._records: Dict[str, ImageRecord] = {}
        self.classes: List[str] = list(classes or [])
        self.settings: Dict[str, Any] = {}
        self._load_project_file()
        self.rescan()

    # ------------------------------------------------------------------ setup

    @property
    def config_path(self) -> Path:
        return self.project_dir / "project.json"

    def _load_project_file(self) -> None:
        if not self.config_path.exists():
            return
        try:
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        stored = [str(c) for c in data.get("classes", [])]
        # Classes passed to the constructor win, but never drop stored ones.
        for name in stored:
            if name not in self.classes:
                self.classes.append(name)
        self.settings.update(data.get("settings") or {})

    def save_project_file(self) -> None:
        _atomic_write_json(
            self.config_path, {"classes": self.classes, "settings": self.settings}
        )

    def rescan(self) -> List[str]:
        """Pick up images added to the folder since the last scan."""
        with self._lock:
            found = []
            for path in sorted(self.image_root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if PROJECT_DIRNAME in path.parts:
                    continue
                rel = path.relative_to(self.image_root).as_posix()
                found.append(rel)
                if rel not in self._records:
                    self._records[rel] = self._load_record(rel)
            for rel in set(self._records) - set(found):
                del self._records[rel]
            return found

    # ------------------------------------------------------------- record I/O

    def _sidecar_path(self, relpath: str) -> Path:
        return self.annotation_dir / (_slug(relpath) + ".json")

    def _load_record(self, relpath: str) -> ImageRecord:
        sidecar = self._sidecar_path(relpath)
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                record = ImageRecord.from_dict(data)
                record.path = relpath  # trust the filesystem over the sidecar
                if not record.width or not record.height:
                    record.width, record.height = image_size(self.abs_path(relpath))
                return record
            except (json.JSONDecodeError, OSError, KeyError):
                pass  # corrupt sidecar: fall through and start this image fresh
        width, height = image_size(self.abs_path(relpath))
        return ImageRecord(path=relpath, width=width, height=height)

    def _persist(self, record: ImageRecord) -> None:
        _atomic_write_json(self._sidecar_path(record.path), record.to_dict())

    # -------------------------------------------------------------- accessors

    def abs_path(self, relpath: str) -> Path:
        """Resolve a project-relative path, refusing anything outside the root."""
        candidate = (self.image_root / relpath).resolve()
        if candidate != self.image_root and self.image_root not in candidate.parents:
            raise ValueError(f"path escapes the project root: {relpath}")
        return candidate

    def __len__(self) -> int:
        return len(self._records)

    def paths(self) -> List[str]:
        with self._lock:
            return sorted(self._records)

    def records(self) -> List[ImageRecord]:
        with self._lock:
            return [self._records[p] for p in sorted(self._records)]

    def get(self, relpath: str) -> ImageRecord:
        with self._lock:
            try:
                return self._records[relpath]
            except KeyError:
                raise KeyError(f"unknown image: {relpath}") from None

    def annotated_records(self) -> List[ImageRecord]:
        return [r for r in self.records() if r.annotations]

    # ---------------------------------------------------------------- mutation

    def set_annotations(
        self,
        relpath: str,
        annotations: Iterable[Annotation],
        status: Optional[str] = None,
    ) -> ImageRecord:
        with self._lock:
            record = self.get(relpath)
            record.annotations = list(annotations)
            if status is not None:
                record.status = status
            record.updated_at = time.time()
            self.register_classes(a.label for a in record.annotations)
            self._persist(record)
            return record

    def set_status(self, relpath: str, status: str) -> ImageRecord:
        if status not in (STATUS_NEW, STATUS_PREDICTED, STATUS_REVIEWED):
            raise ValueError(f"unknown status: {status}")
        with self._lock:
            record = self.get(relpath)
            record.status = status
            record.updated_at = time.time()
            self._persist(record)
            return record

    def register_classes(self, labels: Iterable[str]) -> None:
        """Add unseen labels to the class list, keeping insertion order."""
        with self._lock:
            changed = False
            for label in labels:
                if label and label not in self.classes:
                    self.classes.append(label)
                    changed = True
            if changed:
                self.save_project_file()

    def set_classes(self, classes: Iterable[str]) -> List[str]:
        with self._lock:
            seen: List[str] = []
            for name in classes:
                name = str(name).strip()
                if name and name not in seen:
                    seen.append(name)
            self.classes = seen
            self.save_project_file()
            return self.classes

    def stats(self) -> Dict[str, Any]:
        records = self.records()
        by_status = {STATUS_NEW: 0, STATUS_PREDICTED: 0, STATUS_REVIEWED: 0}
        per_class: Dict[str, int] = {}
        boxes = 0
        for record in records:
            by_status[record.status] = by_status.get(record.status, 0) + 1
            boxes += len(record.annotations)
            for annotation in record.annotations:
                per_class[annotation.label] = per_class.get(annotation.label, 0) + 1
        return {
            "images": len(records),
            "boxes": boxes,
            "by_status": by_status,
            "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        }
