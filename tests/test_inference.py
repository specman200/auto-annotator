import numpy as np
import pytest

from auto_annotator.inference import BACKENDS, load_detector, parse_spec
from auto_annotator.inference.base import nms
from auto_annotator.inference.onnx_yolo import OnnxYoloDetector, letterbox
from auto_annotator.schema import Box


def test_parse_spec_handles_prefixes_and_bare_paths():
    assert parse_spec("mock") == ("mock", None)
    assert parse_spec("onnx:/models/y.onnx") == ("onnx", "/models/y.onnx")
    assert parse_spec("weights/yolov8n.onnx") == ("onnx", "weights/yolov8n.onnx")
    assert parse_spec("yolov8n.pt") == ("ultralytics", "yolov8n.pt")


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError):
        parse_spec("magic:model.bin")


def test_every_registered_backend_can_be_named():
    assert set(BACKENDS) >= {"mock", "onnx", "ultralytics", "torchvision"}


def test_mock_detector_is_deterministic(image_dir):
    detector = load_detector("mock")
    first = detector.predict(image_dir / "a.jpg", conf=0.0)
    second = load_detector("mock").predict(image_dir / "a.jpg", conf=0.0)
    assert [(d.label, d.score) for d in first] == [(d.label, d.score) for d in second]
    assert all(0.0 <= d.box.x1 <= 1.0 for d in first)


def test_predict_applies_the_confidence_floor(image_dir):
    detector = load_detector("mock")
    assert all(d.score >= 0.9 for d in detector.predict(image_dir / "a.jpg", conf=0.9))


def test_onnx_backend_requires_weights():
    with pytest.raises(ValueError):
        load_detector("onnx")


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = [Box(0, 0, 0.5, 0.5), Box(0.01, 0.01, 0.51, 0.51), Box(0.8, 0.8, 1, 1)]
    keep = nms(boxes, [0.6, 0.9, 0.7], iou_threshold=0.45)
    assert sorted(keep) == [1, 2]


def test_letterbox_pads_to_a_square_without_distorting():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = letterbox(image, 64)
    assert canvas.shape == (64, 64, 3)
    assert scale == pytest.approx(0.64)
    assert (pad_x, pad_y) == (0, 16)
    assert canvas[0, 0].tolist() == [114, 114, 114]  # padding colour


def _detector(labels):
    detector = OnnxYoloDetector(weights="unused.onnx", labels=labels)
    return detector


def test_decode_reads_v8_layout():
    """v8 exports are (1, 4 + nc, boxes) with no objectness channel."""
    detector = _detector(["cat", "dog"])
    raw = np.zeros((1, 6, 3), dtype=np.float32)
    raw[0, :4, 0] = [10, 20, 30, 40]
    raw[0, 4:, 0] = [0.9, 0.1]           # strong cat
    raw[0, 4:, 1] = [0.05, 0.05]         # below threshold
    raw[0, :4, 2] = [50, 60, 10, 10]
    raw[0, 4:, 2] = [0.2, 0.8]           # strong dog
    boxes, scores, class_ids = detector._decode(raw, conf=0.5)
    assert class_ids == [0, 1]
    assert scores == pytest.approx([0.9, 0.8])
    assert boxes[0].tolist() == [10, 20, 30, 40]


def test_decode_reads_v5_layout_with_objectness():
    """v5 exports are (1, boxes, 5 + nc); class scores multiply objectness."""
    detector = _detector(["cat", "dog"])
    raw = np.zeros((1, 2, 7), dtype=np.float32)
    raw[0, 0] = [10, 20, 30, 40, 0.5, 0.8, 0.1]   # 0.5 * 0.8 = 0.4
    raw[0, 1] = [50, 60, 10, 10, 1.0, 0.1, 0.95]
    boxes, scores, class_ids = detector._decode(raw, conf=0.3)
    assert class_ids == [0, 1]
    assert scores == pytest.approx([0.4, 0.95])


def test_missing_onnx_file_raises_a_clear_error(tmp_path):
    detector = OnnxYoloDetector(weights=str(tmp_path / "absent.onnx"))
    with pytest.raises(FileNotFoundError):
        detector.load()


def test_detector_describe_exposes_kind_and_labels():
    described = load_detector("mock").describe()
    assert described["kind"] == "mock" and "person" in described["labels"]


def test_decode_orients_both_axis_layouts_the_same_way():
    """The same predictions transposed must decode identically."""
    detector = _detector(["cat", "dog"])
    channels_first = np.zeros((1, 6, 3), dtype=np.float32)
    channels_first[0, :4, 0] = [10, 20, 30, 40]
    channels_first[0, 4:, 0] = [0.9, 0.1]
    boxes_a, scores_a, ids_a = detector._decode(channels_first, conf=0.5)
    boxes_b, scores_b, ids_b = detector._decode(
        np.transpose(channels_first, (0, 2, 1)), conf=0.5
    )
    assert ids_a == ids_b and scores_a == pytest.approx(scores_b)
    assert boxes_a.tolist() == boxes_b.tolist()


def test_decode_orients_by_shape_when_the_labels_do_not_match():
    """Channels cannot be fewer than 5, so the short axis must be the boxes."""
    detector = OnnxYoloDetector(weights="unused.onnx")  # 80 COCO labels
    raw = np.zeros((1, 6, 2), dtype=np.float32)          # 6 channels, 2 boxes
    raw[0, :4, 0] = [1, 2, 3, 4]
    raw[0, 4:, 0] = [0.9, 0.1]
    boxes, scores, class_ids = detector._decode(raw, conf=0.5)
    assert scores == pytest.approx([0.9]) and boxes[0].tolist() == [1, 2, 3, 4]


def test_decode_handles_an_all_zero_tensor():
    detector = OnnxYoloDetector(weights="unused.onnx", labels=["cat"])
    boxes, scores, class_ids = detector._decode(np.zeros((1, 5, 3), dtype=np.float32), conf=0.5)
    assert len(boxes) == 0 and scores == [] and class_ids == []
