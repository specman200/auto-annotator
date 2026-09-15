"""Backend registry: turn a ``kind:weights`` string into a loaded detector."""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import Detector, nms
from .mock import MockDetector

BACKENDS: Dict[str, str] = {
    "mock": "auto_annotator.inference.mock:MockDetector",
    "onnx": "auto_annotator.inference.onnx_yolo:OnnxYoloDetector",
    "openvino": "auto_annotator.inference.openvino_backend:OpenVinoDetector",
    "ultralytics": "auto_annotator.inference.ultralytics_backend:UltralyticsDetector",
    "torchvision": "auto_annotator.inference.torchvision_backend:TorchvisionDetector",
}

BACKEND_HELP = {
    "mock": "deterministic fake boxes; no weights needed (good for trying the GUI)",
    "onnx": "YOLOv5/v8/v11 ONNX export via onnxruntime, e.g. onnx:yolov8n.onnx",
    "openvino": "YOLO via the OpenVINO runtime (CPU/GPU/NPU), "
                "e.g. openvino:yolo11n_openvino_model",
    "ultralytics": "ultralytics YOLO weights, e.g. ultralytics:yolov8n.pt",
    "torchvision": "torchvision COCO detector, e.g. torchvision:fasterrcnn_resnet50_fpn",
}


def _import(target: str) -> Type[Detector]:
    module_name, class_name = target.split(":")
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def parse_spec(spec: str):
    """Split ``"onnx:/models/yolov8n.onnx"`` into ``("onnx", "/models/yolov8n.onnx")``.

    A bare path is treated as the backend that matches its extension, so
    ``--model yolov8n.onnx`` works without a prefix.
    """
    spec = (spec or "mock").strip()
    kind, _, weights = spec.partition(":")
    if kind in BACKENDS:
        return kind, (weights or None)
    lowered = spec.lower()
    if lowered.endswith(".onnx"):
        return "onnx", spec
    if lowered.endswith((".pt", ".pth")):
        return "ultralytics", spec
    # An OpenVINO IR, or the export directory ultralytics writes around it.
    if lowered.endswith(".xml") or lowered.endswith("_openvino_model"):
        return "openvino", spec
    raise ValueError(
        f"unknown model spec {spec!r}. Use one of: "
        + ", ".join(f"{k} ({v})" for k, v in BACKEND_HELP.items())
    )


def load_detector(spec: str = "mock", **options: Any) -> Detector:
    """Build (but do not yet load) the detector named by ``spec``."""
    kind, weights = parse_spec(spec)
    detector_cls = _import(BACKENDS[kind])
    return detector_cls(weights=weights, **options)


__all__ = [
    "BACKENDS",
    "BACKEND_HELP",
    "Detector",
    "MockDetector",
    "load_detector",
    "nms",
    "parse_spec",
]
