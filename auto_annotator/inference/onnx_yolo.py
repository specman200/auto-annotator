"""YOLO-family ONNX models through onnxruntime (CPU by default).

Handles both common export layouts:

* YOLOv8/v11 ``(1, 4 + num_classes, num_boxes)`` — no objectness channel.
* YOLOv5/v7   ``(1, num_boxes, 5 + num_classes)`` — objectness in column 4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..schema import Box, Detection
from .base import Detector, nms
from .coco import COCO_CLASSES

log = logging.getLogger(__name__)


def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio and pad to a square canvas.

    Returns the canvas plus the scale and padding needed to map predictions
    back onto the original image.
    """
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = np.array(
        _pil_resize(image, new_w, new_h), dtype=np.uint8
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _pil_resize(image: np.ndarray, width: int, height: int):
    from PIL import Image

    return Image.fromarray(image).resize((width, height), Image.BILINEAR)


class OnnxYoloDetector(Detector):
    kind = "onnx"

    def __init__(
        self,
        weights: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        imgsz: int = 640,
        iou: float = 0.45,
        **options: Any,
    ):
        super().__init__(weights=weights, **options)
        if not weights:
            raise ValueError(
                "the onnx backend needs a model file, e.g. --model onnx:yolov8n.onnx"
            )
        self.imgsz = int(imgsz)
        self.iou = float(iou)
        self._labels: List[str] = list(labels) if labels else []
        self._explicit_labels = bool(labels)
        self._num_classes: Optional[int] = None
        self._warned_about_labels = False
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
        # Static-shaped exports tell us the size they want; dynamic ones don't.
        if isinstance(shape[-1], int) and shape[-1] > 0:
            self.imgsz = int(shape[-1])
        if not self._labels:
            self._labels = list(self._labels_from_metadata() or COCO_CLASSES)
        self._loaded = True

    def _labels_from_metadata(self) -> Optional[List[str]]:
        """Ultralytics stores ``names`` in the ONNX metadata; use it when present."""
        try:
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if not raw:
                return None
            import ast

            names = ast.literal_eval(raw)
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=int)]
            if isinstance(names, (list, tuple)):
                return [str(n) for n in names]
        except Exception:
            return None
        return None

    @property
    def labels(self) -> Sequence[str]:
        return self._labels

    # -------------------------------------------------------------- inference

    def _predict(self, image_path: Path, conf: float) -> List[Detection]:
        from PIL import Image

        with Image.open(image_path) as handle:
            image = np.array(handle.convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        canvas, scale, pad_x, pad_y = letterbox(image, self.imgsz)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]  # NCHW

        outputs = self._session.run(None, {self._input_name: tensor})
        boxes, scores, class_ids = self._decode(np.asarray(outputs[0]), conf)

        detections: List[Detection] = []
        keep_boxes: List[Box] = []
        for (cx, cy, w, h) in boxes:
            # letterbox space -> original pixels
            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale
            keep_boxes.append(Box.from_pixels((x1, y1, x2, y2), orig_w, orig_h))

        for index in nms(keep_boxes, scores, self.iou):
            detections.append(
                Detection(
                    label=self._label_for(class_ids[index]),
                    box=keep_boxes[index],
                    score=float(scores[index]),
                )
            )
        return detections

    def _label_for(self, class_id: int) -> str:
        """Name a class, but only when the label list actually fits the model.

        A 3-class custom export paired with the default COCO names would
        otherwise come back labelled "person"/"bicycle"; generic names are
        wrong in a way the user can see and fix with ``labels=``.
        """
        if self._num_classes is not None and len(self._labels) != self._num_classes:
            if not self._warned_about_labels:
                log.warning(
                    "%s predicts %d classes but %d label names are configured; "
                    "falling back to class_<n>. Pass labels= to name them.",
                    self.weights, self._num_classes, len(self._labels),
                )
                self._warned_about_labels = True
            return f"class_{class_id}"
        if 0 <= class_id < len(self._labels):
            return self._labels[class_id]
        return f"class_{class_id}"

    def _decode(self, raw: np.ndarray, conf: float):
        """Normalize the two YOLO output layouts into (xywh, score, class)."""
        pred = self._orient(raw[0] if raw.ndim == 3 else raw)

        if pred.shape[1] >= 6 and self._looks_like_v5(pred):
            objectness = pred[:, 4]
            class_scores = pred[:, 5:] * objectness[:, None]
        else:
            class_scores = pred[:, 4:]

        self._num_classes = class_scores.shape[1]
        if class_scores.size == 0:
            return np.empty((0, 4), dtype=np.float32), [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        mask = scores >= conf
        return pred[mask, :4], scores[mask].tolist(), class_ids[mask].tolist()

    def _orient(self, pred: np.ndarray) -> np.ndarray:
        """Return the prediction as (detections, channels).

        v8 exports are channels-first, v5 exports are detections-first, and
        neither says which it is, so match the axis against the class count
        we know the model has and fall back to "the channel axis is the
        smaller one" when the labels are unknown.
        """
        if self._labels:
            expected = {len(self._labels) + 4, len(self._labels) + 5}
            if pred.shape[1] in expected:
                return pred
            if pred.shape[0] in expected:
                return pred.T
        # A channel axis holds 4 box coordinates plus at least one class, so an
        # axis shorter than that cannot be the channels.
        if pred.shape[1] < 5 <= pred.shape[0]:
            return pred.T
        return pred if pred.shape[0] >= pred.shape[1] else pred.T

    def _looks_like_v5(self, pred: np.ndarray) -> bool:
        """v5 has one extra column beyond 4 + len(labels)."""
        if not self._labels:
            return False
        return pred.shape[1] == len(self._labels) + 5
