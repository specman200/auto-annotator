"""Run the OpenVINO backend against a real compiled model.

The IR is converted on the fly from the same synthetic YOLO graph the ONNX
tests use, so this exercises loading, compiling, inference and the shared
decode without downloading weights.
"""

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
pytest.importorskip("onnx")

from auto_annotator.engine import Annotator
from auto_annotator.inference import load_detector, parse_spec
from auto_annotator.inference.openvino_backend import (
    OpenVinoDetector,
    resolve_model_path,
)

from test_onnx_end_to_end import IMGSZ, LABELS, build_model


@pytest.fixture
def wide_image(tmp_path):
    from PIL import Image

    path = tmp_path / "wide.jpg"
    Image.new("RGB", (200, 100), "white").save(path)
    return path


def build_ir(tmp_path, detections, name="fake", layout="v8"):
    """Convert the synthetic ONNX model into an ultralytics-style export dir."""
    onnx_path = build_model(tmp_path / f"{name}.onnx", detections, layout)
    export_dir = tmp_path / f"{name}_openvino_model"
    export_dir.mkdir(exist_ok=True)
    # save_model compresses constants to FP16 by default, which would show up
    # as small score differences; keep the test IR lossless.
    ov.save_model(
        ov.convert_model(str(onnx_path)),
        str(export_dir / f"{name}.xml"),
        compress_to_fp16=False,
    )
    return export_dir


@pytest.mark.parametrize("layout", ["v8", "v5"])
def test_detection_maps_back_to_original_image_coordinates(tmp_path, wide_image, layout):
    detections = [[IMGSZ / 2, IMGSZ / 2, IMGSZ / 2, IMGSZ / 4, 0.9, 0.05]]
    export_dir = build_ir(tmp_path, detections, name=layout, layout=layout)

    detector = OpenVinoDetector(weights=str(export_dir), labels=LABELS, device="CPU")
    result = detector.predict(wide_image, conf=0.5)

    assert len(result) == 1
    assert result[0].label == "cat"
    assert result[0].score == pytest.approx(0.9, abs=1e-5)
    x1, y1, x2, y2 = result[0].box.to_pixels(200, 100)
    assert (x1, x2) == pytest.approx((50.0, 150.0), abs=0.5)
    assert (y1, y2) == pytest.approx((25.0, 75.0), abs=0.5)


def test_openvino_and_onnx_agree_on_the_same_model(tmp_path, wide_image):
    """The two runtimes share their maths; they must share their answers too."""
    from auto_annotator.inference.onnx_yolo import OnnxYoloDetector

    detections = [[32, 32, 20, 16, 0.91, 0.02], [12, 40, 10, 10, 0.64, 0.10]]
    onnx_path = build_model(tmp_path / "shared.onnx", detections)
    export_dir = build_ir(tmp_path, detections, name="shared2")

    def run(detector):
        return [
            (d.label, d.score, d.box.to_pixels(200, 100))
            for d in detector.predict(wide_image, conf=0.5)
        ]

    from_openvino = run(OpenVinoDetector(weights=str(export_dir), labels=LABELS))
    from_onnx = run(OnnxYoloDetector(weights=str(onnx_path), labels=LABELS))

    assert len(from_openvino) == len(from_onnx)
    for (label, score, box), (other_label, other_score, other_box) in zip(
        from_openvino, from_onnx
    ):
        assert label == other_label
        assert score == pytest.approx(other_score, abs=1e-4)
        assert box == pytest.approx(other_box, abs=1e-3)


def test_input_size_is_read_from_the_compiled_model(tmp_path):
    export_dir = build_ir(tmp_path, [[0, 0, 1, 1, 0.9, 0.0]], name="size")
    detector = OpenVinoDetector(weights=str(export_dir), labels=LABELS, imgsz=1234)
    detector.load()
    assert detector.imgsz == IMGSZ


def test_it_reads_an_onnx_file_directly(tmp_path, wide_image):
    """OpenVINO can consume ONNX, which saves a conversion step."""
    onnx_path = build_model(tmp_path / "direct.onnx", [[32, 32, 20, 20, 0.9, 0.0]])
    detector = OpenVinoDetector(weights=str(onnx_path), labels=LABELS)
    assert len(detector.predict(wide_image, conf=0.5)) == 1


def test_class_names_come_from_metadata_yaml(tmp_path, wide_image):
    export_dir = build_ir(tmp_path, [[32, 32, 20, 20, 0.1, 0.9]], name="meta")
    (export_dir / "metadata.yaml").write_text(
        "description: fake\nnames:\n  0: bolt\n  1: washer\n", encoding="utf-8"
    )
    detector = OpenVinoDetector(weights=str(export_dir))  # no labels passed
    assert [d.label for d in detector.predict(wide_image, conf=0.5)] == ["washer"]


def test_explicit_labels_beat_the_metadata(tmp_path, wide_image):
    export_dir = build_ir(tmp_path, [[32, 32, 20, 20, 0.9, 0.1]], name="override")
    (export_dir / "metadata.yaml").write_text("names:\n  0: bolt\n  1: washer\n")
    detector = OpenVinoDetector(weights=str(export_dir), labels=["cat", "dog"])
    assert [d.label for d in detector.predict(wide_image, conf=0.5)] == ["cat"]


def test_a_directory_without_an_ir_is_a_clear_error(tmp_path):
    (tmp_path / "empty_openvino_model").mkdir()
    with pytest.raises(FileNotFoundError, match="no OpenVINO .xml"):
        resolve_model_path(str(tmp_path / "empty_openvino_model"))
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path / "absent.xml"))


def test_spec_parsing_recognises_openvino_paths():
    assert parse_spec("openvino:yolo11n_openvino_model") == (
        "openvino", "yolo11n_openvino_model",
    )
    assert parse_spec("weights/yolo11n_openvino_model")[0] == "openvino"
    assert parse_spec("weights/yolo11n.xml") == ("openvino", "weights/yolo11n.xml")


def test_device_is_reported_and_configurable(tmp_path):
    export_dir = build_ir(tmp_path, [[0, 0, 1, 1, 0.9, 0.0]], name="device")
    detector = OpenVinoDetector(weights=str(export_dir), device="CPU")
    described = detector.describe()
    assert described["kind"] == "openvino" and described["device"] == "CPU"


def test_openvino_backend_drives_a_whole_project(tmp_path, project):
    export_dir = build_ir(tmp_path, [[32, 32, 16, 16, 0.95, 0.0]], name="project")
    detector = load_detector(f"openvino:{export_dir}", labels=LABELS)
    result = Annotator(project, detector=detector, conf=0.5).annotate_all()
    assert result["processed"] == 3 and result["boxes"] == 3
    assert project.get("a.jpg").annotations[0].label == "cat"
