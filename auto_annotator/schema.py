"""Core data types for annotations, images and projects.

Boxes are stored in *normalized* xyxy form (0..1 relative to image width/height)
so that annotations survive resizing or re-encoding the source images.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# How an annotation came to exist. The GUI colours boxes by this.
SOURCE_MODEL = "model"
SOURCE_HUMAN = "human"

# Review state of a whole image.
STATUS_NEW = "new"  # never touched
STATUS_PREDICTED = "predicted"  # model ran, nobody checked it
STATUS_REVIEWED = "reviewed"  # a human signed off


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


@dataclass
class Box:
    """Axis-aligned box in normalized xyxy coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        self.x1, self.x2 = sorted((_clamp01(self.x1), _clamp01(self.x2)))
        self.y1, self.y2 = sorted((_clamp01(self.y1), _clamp01(self.y2)))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixels(self, image_width: int, image_height: int) -> List[float]:
        return [
            self.x1 * image_width,
            self.y1 * image_height,
            self.x2 * image_width,
            self.y2 * image_height,
        ]

    @classmethod
    def from_pixels(
        cls, xyxy, image_width: int, image_height: int
    ) -> "Box":
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        return cls(
            x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height
        )

    def iou(self, other: "Box") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Box":
        return cls(
            float(data["x1"]), float(data["y1"]), float(data["x2"]), float(data["y2"])
        )


@dataclass
class Annotation:
    """One labelled object on one image."""

    label: str
    box: Box
    score: Optional[float] = None
    source: str = SOURCE_HUMAN
    id: str = field(default_factory=_new_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "box": self.box.to_dict(),
            "score": self.score,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        score = data.get("score")
        return cls(
            id=str(data.get("id") or _new_id()),
            label=str(data["label"]),
            box=Box.from_dict(data["box"]),
            score=None if score is None else float(score),
            source=str(data.get("source") or SOURCE_HUMAN),
        )


@dataclass
class ImageRecord:
    """An image in the project plus everything known about it."""

    path: str  # relative to the project's image root, posix separators
    width: int = 0
    height: int = 0
    status: str = STATUS_NEW
    annotations: List[Annotation] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "annotations": [a.to_dict() for a in self.annotations],
            "updated_at": self.updated_at,
        }

    def summary(self) -> Dict[str, Any]:
        """Light-weight form for the image list in the sidebar."""
        return {
            "path": self.path,
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "status": self.status,
            "count": len(self.annotations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRecord":
        return cls(
            path=str(data["path"]),
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            status=str(data.get("status") or STATUS_NEW),
            annotations=[Annotation.from_dict(a) for a in data.get("annotations", [])],
            updated_at=float(data.get("updated_at") or time.time()),
        )


@dataclass
class Detection:
    """Raw model output before it becomes an annotation."""

    label: str
    box: Box
    score: float

    def to_annotation(self) -> Annotation:
        return Annotation(
            label=self.label, box=self.box, score=self.score, source=SOURCE_MODEL
        )
