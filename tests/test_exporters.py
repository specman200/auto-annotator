import csv
import json
from xml.etree import ElementTree as ET

import pytest

from auto_annotator import exporters
from auto_annotator.schema import Annotation, Box


@pytest.fixture
def labelled(project):
    project.set_annotations("a.jpg", [  # 200x100
        Annotation("cat", Box(0.1, 0.2, 0.6, 0.7), 0.88, "model"),
        Annotation("dog", Box(0.0, 0.0, 0.5, 0.5), None, "human"),
    ])
    return project


def test_coco_boxes_are_pixel_xywh(labelled, tmp_path):
    path = exporters.export_coco(labelled, tmp_path / "out.json")
    data = json.loads(path.read_text())
    assert [image["file_name"] for image in data["images"]] == ["a.jpg"]
    assert data["annotations"][0]["bbox"] == [20.0, 20.0, 100.0, 50.0]
    assert data["annotations"][0]["area"] == 5000.0
    names = {category["id"]: category["name"] for category in data["categories"]}
    assert names[data["annotations"][0]["category_id"]] == "cat"


def test_coco_can_include_empty_images(labelled, tmp_path):
    path = exporters.export_coco(labelled, tmp_path / "all.json", only_annotated=False)
    assert len(json.loads(path.read_text())["images"]) == 3


def test_yolo_writes_normalized_centre_boxes_and_class_files(labelled, tmp_path):
    out = exporters.export_yolo(labelled, tmp_path / "yolo")
    lines = (out / "labels" / "a.txt").read_text().strip().splitlines()
    class_id, cx, cy, w, h = lines[0].split()
    assert class_id == str(labelled.classes.index("cat"))
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.35, 0.45, 0.5, 0.5))
    assert "cat" in (out / "classes.txt").read_text()
    assert "names:" in (out / "data.yaml").read_text()


def test_voc_writes_one_xml_per_image(labelled, tmp_path):
    out = exporters.export_voc(labelled, tmp_path / "voc")
    root = ET.parse(out / "a.xml").getroot()
    assert root.findtext("filename") == "a.jpg"
    assert root.findtext("size/width") == "200"
    box = root.find("object/bndbox")
    assert [box.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")] == ["20", "20", "120", "70"]


def test_csv_has_one_row_per_box(labelled, tmp_path):
    path = exporters.export_csv(labelled, tmp_path / "out.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["label"] == "cat" and rows[0]["source"] == "model"
    assert rows[1]["score"] == ""


def test_export_dispatch_defaults_into_the_project_folder(labelled):
    for fmt in exporters.FORMATS:
        path = exporters.export(labelled, fmt)
        assert path.exists()
        assert labelled.project_dir in path.parents


def test_unknown_format_is_rejected(labelled):
    with pytest.raises(ValueError):
        exporters.export(labelled, "pascal-voc-2007")


def test_labels_not_in_the_class_list_still_export(project, tmp_path):
    project.set_classes([])
    project.set_annotations("a.jpg", [Annotation("surprise", Box(0, 0, 1, 1))])
    data = json.loads(exporters.export_coco(project, tmp_path / "o.json").read_text())
    assert [category["name"] for category in data["categories"]] == ["surprise"]
