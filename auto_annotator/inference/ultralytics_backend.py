"""Ultralytics YOLO (.pt) weights, run locally through the ultralytics package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from ..schema import Box, Detection
from .base import Detector


class UltralyticsDetector(Detector):
    kind = "ultralytics"

    def __init__(
        self,
        weights: Optional[str] = None,
        imgsz: int = 640,
        device: str = "cpu",
        iou: float = 0.45,
        **options: Any,
    ):
        super().__init__(weights=weights or "yolov8n.pt", **options)
        self.imgsz = int(imgsz)
        self.device = device
        self.iou = float(iou)
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO  # optional dependency

        self._model = YOLO(self.weights)
        self._loaded = True

    @property
    def labels(self) -> Sequence[str]:
        if self._model is None:
            return []
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        return list(names)

    def _predict(self, image_path: Path, conf: float) -> List[Detection]:
        results = self._model.predict(
            source=str(image_path),
            conf=conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            height, width = result.orig_shape
            names = result.names
            for row in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in row.xyxy[0].tolist())
                class_id = int(row.cls[0])
                detections.append(
                    Detection(
                        label=str(names.get(class_id, f"class_{class_id}")),
                        box=Box.from_pixels((x1, y1, x2, y2), width, height),
                        score=float(row.conf[0]),
                    )
                )
        return detections
