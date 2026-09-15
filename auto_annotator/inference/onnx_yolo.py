"""YOLO-family ONNX models through onnxruntime (CPU by default).

Handles both common export layouts:

* YOLOv8/v11 ``(1, 4 + num_classes, num_boxes)`` — no objectness channel.
* YOLOv5/v7   ``(1, num_boxes, 5 + num_classes)`` — objectness in column 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .yolo_common import YoloRuntime, letterbox  # letterbox re-exported

__all__ = ["OnnxYoloDetector", "letterbox"]


class OnnxYoloDetector(YoloRuntime):
    kind = "onnx"
    example_weights = "yolov8n.onnx"

    def __init__(self, weights: Optional[str] = None, **options: Any):
        super().__init__(weights=weights, **options)
        self._session = None

    # ------------------------------------------------------------- lifecycle

    def load(self) -> None:
        import onnxruntime as ort  # imported here so the dep stays optional

        path = Path(self.weights).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        providers = self.options.get("providers") or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(path), providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name

        shape = self._session.get_inputs()[0].shape
        # Static-shaped exports tell us the size they want; dynamic ones don\'t.
        if isinstance(shape[-1], int) and shape[-1] > 0:
            self.imgsz = int(shape[-1])
        if not self._explicit_labels:
            self._labels = self._labels_from_metadata() or self._default_labels()
        self._loaded = True

    def _labels_from_metadata(self) -> Optional[List[str]]:
        """Ultralytics stores ``names`` in the ONNX metadata; use it when present."""
        try:
            meta = self._session.get_modelmeta().custom_metadata_map or {}
        except Exception:
            return None
        raw = meta.get("names")
        return self._names_from_mapping(raw) if raw else None

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        return self._session.run(None, {self._input_name: tensor})[0]
