import pytest

from auto_annotator.schema import Annotation, Box, Detection, ImageRecord


def test_box_normalizes_corner_order_and_clamps():
    box = Box(0.8, 1.4, 0.2, -0.3)
    assert (box.x1, box.x2) == (0.2, 0.8)
    assert (box.y1, box.y2) == (0.0, 1.0)


def test_box_pixel_round_trip():
    box = Box.from_pixels((10, 20, 110, 70), 200, 100)
    assert box.to_pixels(200, 100) == pytest.approx([10, 20, 110, 70])
    assert box.width == 0.5


def test_iou():
    assert Box(0, 0, 1, 1).iou(Box(0, 0, 1, 1)) == 1.0
    assert Box(0, 0, 0.5, 0.5).iou(Box(0.5, 0.5, 1, 1)) == 0.0
    assert round(Box(0, 0, 0.5, 1).iou(Box(0.25, 0, 0.75, 1)), 3) == 0.333


def test_annotation_round_trip_keeps_id():
    annotation = Annotation("cat", Box(0.1, 0.1, 0.2, 0.2), 0.9, "model")
    clone = Annotation.from_dict(annotation.to_dict())
    assert (clone.id, clone.label, clone.score, clone.source) == (
        annotation.id, "cat", 0.9, "model",
    )


def test_detection_becomes_a_model_annotation():
    annotation = Detection("cat", Box(0, 0, 1, 1), 0.5).to_annotation()
    assert annotation.source == "model" and annotation.score == 0.5


def test_image_record_summary():
    record = ImageRecord("nested/c.png", 64, 64, annotations=[
        Annotation("cat", Box(0, 0, 1, 1))
    ])
    assert record.summary()["name"] == "c.png"
    assert record.summary()["count"] == 1
