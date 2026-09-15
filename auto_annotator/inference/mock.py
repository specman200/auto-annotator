"""A deterministic stand-in detector.

It needs no weights and no downloads, which makes it the backend the tests
and the ``--demo`` walkthrough use: the same image always yields the same
boxes, so the GUI has something to correct.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, List, Optional, Sequence

from ..schema import Box, Detection
from .base import Detector

DEFAULT_LABELS = ["object", "person", "vehicle"]


class MockDetector(Detector):
    kind = "mock"

    def __init__(
        self,
        weights: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        boxes_per_image: int = 2,
        **options: Any,
    ):
        super().__init__(weights=weights, **options)
        self._labels = list(labels) if labels else list(DEFAULT_LABELS)
        self.boxes_per_image = max(1, int(boxes_per_image))

    @property
    def labels(self) -> Sequence[str]:
        return self._labels

    def _predict(self, image_path: Path, conf: float) -> List[Detection]:
        digest = hashlib.sha256(image_path.name.encode("utf-8")).digest()
        detections: List[Detection] = []
        for index in range(self.boxes_per_image):
            offset = index * 6
            x = digest[offset] / 255 * 0.5
            y = digest[offset + 1] / 255 * 0.5
            w = 0.15 + digest[offset + 2] / 255 * 0.3
            h = 0.15 + digest[offset + 3] / 255 * 0.3
            score = 0.5 + digest[offset + 4] / 255 * 0.5
            label = self._labels[digest[offset + 5] % len(self._labels)]
            detections.append(
                Detection(
                    label=label,
                    box=Box(x, y, min(1.0, x + w), min(1.0, y + h)),
                    score=round(score, 3),
                )
            )
        return detections
