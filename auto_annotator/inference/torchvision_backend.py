"""torchvision's pretrained COCO detectors (Faster R-CNN, RetinaNet, FCOS, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

from ..schema import Box, Detection
from .base import Detector
from .coco import TORCHVISION_COCO_CLASSES


class TorchvisionDetector(Detector):
    kind = "torchvision"

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "cpu",
        **options: Any,
    ):
        # ``weights`` names the architecture here, e.g. fasterrcnn_resnet50_fpn.
        super().__init__(weights=weights or "fasterrcnn_resnet50_fpn", **options)
        self.device = device
        self._model = None
        self._labels: List[str] = list(TORCHVISION_COCO_CLASSES)

    def load(self) -> None:
        import torch
        import torchvision

        factory = getattr(torchvision.models.detection, self.weights, None)
        if factory is None:
            raise ValueError(
                f"unknown torchvision detector: {self.weights}. "
                "Try fasterrcnn_resnet50_fpn or retinanet_resnet50_fpn."
            )
        self._model = factory(weights="DEFAULT")
        self._model.eval().to(self.device)
        self._torch = torch
        self._loaded = True

    @property
    def labels(self) -> Sequence[str]:
        return [name for name in self._labels if name not in ("N/A", "__background__")]

    def _predict(self, image_path: Path, conf: float) -> List[Detection]:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as handle:
            image = np.array(handle.convert("RGB"))
        height, width = image.shape[:2]
        tensor = (
            self._torch.from_numpy(image).permute(2, 0, 1).float().div(255).to(self.device)
        )
        with self._torch.inference_mode():
            output = self._model([tensor])[0]

        detections: List[Detection] = []
        for box, score, label_id in zip(
            output["boxes"].tolist(), output["scores"].tolist(), output["labels"].tolist()
        ):
            if score < conf:
                continue
            index = int(label_id)
            name = (
                self._labels[index]
                if 0 <= index < len(self._labels)
                else f"class_{index}"
            )
            detections.append(
                Detection(
                    label=name,
                    box=Box.from_pixels(box, width, height),
                    score=float(score),
                )
            )
        return detections
