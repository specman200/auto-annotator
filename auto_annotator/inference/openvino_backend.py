"""YOLO models through the OpenVINO runtime.

Point it at an ultralytics OpenVINO export directory, the ``.xml`` inside it,
or any ONNX file — OpenVINO reads all of them::

    yolo export model=yolo11n.pt format=openvino     # yolo11n_openvino_model/
    auto-annotator serve ./images --model openvino:yolo11n_openvino_model

``device`` picks the target (CPU, GPU, NPU, AUTO, ...); ``auto-annotator
backends`` lists what this machine actually has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .yolo_common import YoloRuntime


class OpenVinoDetector(YoloRuntime):
    kind = "openvino"
    example_weights = "yolo11n_openvino_model"

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "AUTO",
        **options: Any,
    ):
        super().__init__(weights=weights, **options)
        self.device = device
        self._compiled = None
        self._output = None

    # ------------------------------------------------------------- lifecycle

    def load(self) -> None:
        import openvino as ov  # optional dependency

        model_path = resolve_model_path(self.weights)
        core = ov.Core()
        model = core.read_model(model_path)

        self._compiled = core.compile_model(model, self.device)
        self._output = self._compiled.output(0)

        # A static input shape tells us the size the model wants.
        shape = self._compiled.input(0).partial_shape
        if shape.rank.is_static and len(shape) == 4 and shape[3].is_static:
            self.imgsz = shape[3].get_length()

        if not self._explicit_labels:
            self._labels = (
                labels_from_metadata(model_path, model) or self._default_labels()
            )
        self._loaded = True

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        return self._compiled(tensor)[self._output]

    def describe(self) -> dict:
        return {**super().describe(), "device": self.device}


def resolve_model_path(weights: str) -> Path:
    """Accept an export directory, an ``.xml``, or an ONNX file.

    ``yolo export format=openvino`` writes a directory, so pointing at it is
    the natural thing to do; find the IR inside.
    """
    path = Path(weights).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("*.xml"))
        if not candidates:
            raise FileNotFoundError(f"no OpenVINO .xml model inside {path}")
        return candidates[0]
    if not path.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {path}")
    return path


def labels_from_metadata(model_path: Path, model: Any) -> Optional[List[str]]:
    """Read class names from ultralytics' metadata.yaml, or the model itself."""
    names = _names_from_yaml(model_path.parent / "metadata.yaml")
    if names:
        return names
    try:  # ultralytics also stamps names into the IR's runtime info
        raw = model.get_rt_info("model_info", "names")
        return YoloRuntime._names_from_mapping(str(raw.value))
    except Exception:
        return None


def _names_from_yaml(path: Path) -> Optional[List[str]]:
    """Pull ``names`` out of metadata.yaml without requiring a YAML parser."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    try:
        import yaml

        return YoloRuntime._names_from_mapping(yaml.safe_load(text).get("names"))
    except ImportError:
        pass

    # Minimal fallback for the shape ultralytics writes:
    #   names:
    #     0: person
    #     1: bicycle
    names: dict = {}
    in_names = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("names:"):
            inline = line.split(":", 1)[1].strip()
            if inline:  # names: {0: person, 1: bicycle}
                return YoloRuntime._names_from_mapping(inline)
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break
            key, _, value = line.strip().partition(":")
            if value:
                names[key.strip()] = value.strip().strip("'\"")
    return YoloRuntime._names_from_mapping(names) if names else None
