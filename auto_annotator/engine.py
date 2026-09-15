"""Running a detector over a project, one image or all of them."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional

from .inference import Detector, load_detector
from .schema import (
    SOURCE_HUMAN,
    SOURCE_MODEL,
    STATUS_NEW,
    STATUS_PREDICTED,
    STATUS_REVIEWED,
    Annotation,
)
from .store import Project

# What to do with annotations an image already has.
MERGE_REPLACE = "replace"  # drop everything, keep only the new predictions
MERGE_KEEP_HUMAN = "keep-human"  # drop old model boxes, keep hand-drawn ones
MERGE_APPEND = "append"  # keep everything, add the new predictions
MERGE_MODES = (MERGE_REPLACE, MERGE_KEEP_HUMAN, MERGE_APPEND)


class Annotator:
    """Applies a detector's output to a project, honouring the merge policy."""

    def __init__(
        self,
        project: Project,
        detector: Optional[Detector] = None,
        model_spec: str = "mock",
        conf: float = 0.25,
        **detector_options: Any,
    ):
        self.project = project
        self.detector = detector or load_detector(model_spec, **detector_options)
        self.conf = float(conf)

    # ------------------------------------------------------------ single image

    def annotate_image(
        self,
        relpath: str,
        conf: Optional[float] = None,
        merge: str = MERGE_KEEP_HUMAN,
        min_box_area: float = 0.0,
    ) -> List[Annotation]:
        if merge not in MERGE_MODES:
            raise ValueError(f"unknown merge mode: {merge}")

        record = self.project.get(relpath)
        threshold = self.conf if conf is None else float(conf)
        detections = self.detector.predict(
            self.project.abs_path(relpath), conf=threshold
        )
        predicted = [
            d.to_annotation() for d in detections if d.box.area >= min_box_area
        ]

        if merge == MERGE_REPLACE:
            merged = predicted
        elif merge == MERGE_KEEP_HUMAN:
            merged = [a for a in record.annotations if a.source == SOURCE_HUMAN]
            merged += predicted
        else:
            merged = list(record.annotations) + predicted

        status = STATUS_REVIEWED if record.status == STATUS_REVIEWED else STATUS_PREDICTED
        self.project.set_annotations(relpath, merged, status=status)
        return merged

    # ------------------------------------------------------------------ batch

    def annotate_all(
        self,
        paths: Optional[List[str]] = None,
        conf: Optional[float] = None,
        merge: str = MERGE_KEEP_HUMAN,
        skip_reviewed: bool = True,
        skip_annotated: bool = False,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        targets = list(paths if paths is not None else self.project.paths())
        done, boxes, skipped = 0, 0, 0
        errors: List[Dict[str, str]] = []

        for index, relpath in enumerate(targets, start=1):
            record = self.project.get(relpath)
            if skip_reviewed and record.status == STATUS_REVIEWED:
                skipped += 1
            elif skip_annotated and record.annotations:
                skipped += 1
            else:
                try:
                    boxes += len(self.annotate_image(relpath, conf=conf, merge=merge))
                    done += 1
                except Exception as exc:  # one bad image must not kill the batch
                    errors.append({"path": relpath, "error": str(exc)})
            if progress:
                progress(index, len(targets), relpath)

        return {
            "processed": done,
            "skipped": skipped,
            "boxes": boxes,
            "errors": errors,
            "total": len(targets),
        }


class JobRunner:
    """Runs batch annotation on a background thread so the GUI stays responsive."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cancelled: set = set()

    def start(self, annotator: Annotator, **kwargs: Any) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:8]
        total = len(kwargs.get("paths") or annotator.project.paths())
        job = {
            "id": job_id,
            "state": "running",
            "done": 0,
            "total": total,
            "current": None,
            "started_at": time.time(),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        def progress(done: int, count: int, current: str) -> None:
            with self._lock:
                job["done"], job["total"], job["current"] = done, count, current
            if job_id in self._cancelled:
                raise _Cancelled()

        def run() -> None:
            try:
                result = annotator.annotate_all(progress=progress, **kwargs)
                with self._lock:
                    job["result"], job["state"] = result, "done"
            except _Cancelled:
                with self._lock:
                    job["state"] = "cancelled"
            except Exception as exc:
                with self._lock:
                    job["state"] = "error"
                    job["error"] = f"{exc}\n{traceback.format_exc(limit=3)}"
            finally:
                with self._lock:
                    job["finished_at"] = time.time()
                self._cancelled.discard(job_id)

        threading.Thread(target=run, name=f"annotate-{job_id}", daemon=True).start()
        return job

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["state"] != "running":
                return False
        self._cancelled.add(job_id)
        return True


class _Cancelled(Exception):
    """Raised inside the progress callback to unwind a cancelled job."""
