"""Detector interface shared by every local-inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..schema import Box, Detection


class Detector(ABC):
    """A model that turns an image path into detections.

    Implementations load their weights lazily in :meth:`load` so that
    constructing a detector never costs a model load or a download.
    """

    #: short name used in ``--model <name>:<weights>`` specs
    kind = "base"

    def __init__(self, weights: Optional[str] = None, **options: Any):
        self.weights = weights
        self.options = options
        self._loaded = False

    # ------------------------------------------------------------- lifecycle

    def load(self) -> None:
        """Load weights. Called automatically on first use."""
        self._loaded = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # -------------------------------------------------------------- inference

    @abstractmethod
    def _predict(self, image_path: Path, conf: float) -> List[Detection]:
        """Run the model. Return detections in normalized coordinates."""

    def predict(self, image_path, conf: float = 0.25) -> List[Detection]:
        self.ensure_loaded()
        detections = self._predict(Path(image_path), conf)
        return [d for d in detections if d.score >= conf]

    # ------------------------------------------------------------------ meta

    @property
    def labels(self) -> Sequence[str]:
        """Class names the model can emit (may be empty if unknown)."""
        return []

    def describe(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "weights": self.weights,
            "loaded": self._loaded,
            "labels": list(self.labels),
        }


def nms(
    boxes: Sequence[Box], scores: Sequence[float], iou_threshold: float = 0.45
) -> List[int]:
    """Plain greedy non-maximum suppression. Returns indices to keep."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if boxes[best].iou(boxes[i]) < iou_threshold]
    return keep
