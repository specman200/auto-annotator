"""Exercise the real onnxruntime path with a hand-built YOLO-shaped model.

No weights are downloaded: the graph ignores its input and emits fixed
detections, which is enough to prove that session loading, letterbox
pre-processing, decoding and the mapping back to original image
coordinates all line up.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from auto_annotator.engine import Annotator
from auto_annotator.inference.onnx_yolo import OnnxYoloDetector

IMGSZ = 64
LABELS = ["cat", "dog"]


def build_model(path, detections, layout="v8"):
    """Write an ONNX file whose single output is ``detections``.

    ``detections`` is (num_boxes, 4 + len(LABELS)) in v8 style; the file is
    written channel-first for "v8" and box-first with an objectness column
    for "v5", matching what the two export toolchains actually produce.
    """
    from onnx import TensorProto, helper, numpy_helper

    array = np.asarray(detections, dtype=np.float32)
    if layout == "v8":
        payload = array.T[None]  # (1, 4 + nc, boxes)
    else:
        objectness = np.ones((array.shape[0], 1), dtype=np.float32)
        payload = np.hstack([array[:, :4], objectness, array[:, 4:]])[None]

    # The detections are constant, but they are added to a zeroed reduction of
    # the input so the graph genuinely consumes it — a runtime that prunes dead
    # inputs (OpenVINO does) would otherwise end up with no input at all.
    constant = helper.make_node(
        "Constant", [], ["det"], value=numpy_helper.from_array(payload, "det")
    )
    zero = helper.make_node(
        "Constant", [], ["zero"],
        value=numpy_helper.from_array(np.zeros((), dtype=np.float32), "zero"),
    )
    reduced = helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0)
    silenced = helper.make_node("Mul", ["mean", "zero"], ["silenced"])
    add = helper.make_node("Add", ["det", "silenced"], ["output0"])
    graph = helper.make_graph(
        [constant, zero, reduced, silenced, add],
        "fake-yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IMGSZ, IMGSZ])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(payload.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


@pytest.fixture
def wide_image(tmp_path):
    """200x100 — letterboxing must pad it top and bottom."""
    from PIL import Image

    path = tmp_path / "wide.jpg"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


@pytest.mark.parametrize("layout", ["v8", "v5"])
def test_detection_maps_back_to_original_image_coordinates(tmp_path, wide_image, layout):
    # A box centred in the letterboxed canvas, half its width.
    detections = [[IMGSZ / 2, IMGSZ / 2, IMGSZ / 2, IMGSZ / 4, 0.9, 0.05]]
    weights = build_model(tmp_path / f"{layout}.onnx", detections, layout)

    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    result = detector.predict(wide_image, conf=0.5)

    assert len(result) == 1
    detection = result[0]
    assert detection.label == "cat"
    assert detection.score == pytest.approx(0.9, abs=1e-5)

    # scale = 64/200 = 0.32, so the 100px-tall image occupies 32px, padded by 16.
    x1, y1, x2, y2 = detection.box.to_pixels(200, 100)
    assert (x1, x2) == pytest.approx((50.0, 150.0), abs=0.5)
    assert (y1, y2) == pytest.approx((25.0, 75.0), abs=0.5)


def test_input_size_is_read_from_the_model(tmp_path):
    weights = build_model(tmp_path / "m.onnx", [[0, 0, 1, 1, 0.9, 0.0]])
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS, imgsz=1234)
    detector.load()
    assert detector.imgsz == IMGSZ


def test_overlapping_detections_are_suppressed(tmp_path, wide_image):
    detections = [
        [32, 32, 20, 20, 0.9, 0.0],   # best
        [33, 33, 20, 20, 0.7, 0.0],   # nearly the same box
        [10, 10, 8, 8, 0.8, 0.0],     # elsewhere
    ]
    weights = build_model(tmp_path / "nms.onnx", detections)
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    scores = sorted(d.score for d in detector.predict(wide_image, conf=0.5))
    assert scores == pytest.approx([0.8, 0.9])


def test_low_scoring_detections_are_dropped(tmp_path, wide_image):
    weights = build_model(tmp_path / "low.onnx", [[32, 32, 10, 10, 0.3, 0.1]])
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    assert detector.predict(wide_image, conf=0.5) == []
    assert len(detector.predict(wide_image, conf=0.2)) == 1


def test_labels_fall_back_to_coco_when_the_model_has_no_names(tmp_path, wide_image):
    weights = build_model(tmp_path / "coco.onnx", [[32, 32, 10, 10, 0.9, 0.1]])
    detector = OnnxYoloDetector(weights=str(weights))
    detector.load()
    assert detector.labels[0] == "person"


def test_onnx_detector_drives_a_whole_project(tmp_path, project):
    weights = build_model(tmp_path / "project.onnx", [[32, 32, 16, 16, 0.95, 0.0]])
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    result = Annotator(project, detector=detector, conf=0.5).annotate_all()
    assert result["processed"] == 3 and result["boxes"] == 3
    assert project.get("a.jpg").annotations[0].label == "cat"
    assert "cat" in project.classes


def test_a_model_with_no_detections_returns_nothing(tmp_path, wide_image):
    """An all-zero prediction tensor must not blow up on an empty argmax."""
    weights = build_model(tmp_path / "empty.onnx", [[0, 0, 0, 0, 0.0, 0.0]])
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    assert detector.predict(wide_image, conf=0.5) == []


def test_class_count_mismatch_falls_back_to_generic_names(tmp_path, wide_image, caplog):
    """COCO names on a 2-class model would be confidently wrong."""
    weights = build_model(tmp_path / "custom.onnx", [[32, 32, 10, 10, 0.9, 0.1]])
    detector = OnnxYoloDetector(weights=str(weights))  # defaults to 80 COCO names
    result = detector.predict(wide_image, conf=0.5)
    assert [d.label for d in result] == ["class_0"]
    assert "falling back to class_" in caplog.text


def test_orientation_survives_a_model_with_fewer_boxes_than_channels(tmp_path, wide_image):
    """Tiled or single-object exports can emit fewer boxes than channels."""
    weights = build_model(tmp_path / "few.onnx", [[32, 32, 10, 10, 0.9, 0.1]])
    detector = OnnxYoloDetector(weights=str(weights), labels=LABELS)
    assert len(detector.predict(wide_image, conf=0.5)) == 1
