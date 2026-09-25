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
import logging
import os
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import (
    STATUS_NEW,
    STATUS_PREDICTED,
    STATUS_REVIEWED,
    Annotation,
    ImageRecord,
)

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_DIRNAME = ".auto-annotator"


# Waits between attempts to replace a file another process is holding open.
REPLACE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _clear_read_only(path: Path) -> None:
    """Sync clients mark files read-only while they upload; undo that."""
    try:
        if path.exists():
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _save_error(path: Path, cause: OSError) -> OSError:
    """One message for every way a save can be refused, saying what to do."""
    return OSError(
        f"cannot save annotations to {path}: {cause}. Something else is holding "
        f"the file — a sync client (Box, OneDrive, Dropbox) uploading it, or a "
        f"virus scanner. It usually clears in a moment; pausing the sync client "
        f"while you annotate avoids it, as does keeping the images on a local "
        f"disk and exporting to the synced folder when you are done."
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON so an interrupted save cannot leave a half-written file.

    The write goes to a temporary file and is renamed over the target, which
    is atomic on every platform we care about. On Windows that rename fails
    with "access is denied" whenever something else has the target open for a
    moment — a sync client uploading it (Box, OneDrive, Dropbox) or a virus
    scanner reading it. Those locks are brief, so retry; and if the rename
    truly cannot be done, fall back to writing the file in place rather than
    losing the annotations.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)

    # A unique name per write: two saves of the same image at once would
    # otherwise share one temporary file and clobber each other.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
    except OSError as exc:      # the folder itself is locked or read-only
        raise _save_error(path, exc) from exc

    try:
        last_error: Optional[OSError] = None
        for delay in (*REPLACE_DELAYS, None):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:      # Windows: the target is locked
                last_error = exc
                _clear_read_only(path)
            except OSError as exc:
                last_error = exc
            if delay is None:
                break
            time.sleep(delay)

        # The rename will not go through. Writing in place gives up atomicity
        # for this one save, which is a better trade than dropping the work.
        try:
            _clear_read_only(path)
            path.write_text(text, encoding="utf-8")
            log.warning(
                "could not replace %s (%s); wrote it in place instead", path, last_error
            )
            return
        except OSError as exc:
            raise _save_error(path, exc) from exc
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass  # already renamed away, or gone


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
