import csv
import json
import os
from pathlib import Path
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
    lines = (out / "labels" / "train" / "a.txt").read_text().strip().splitlines()
    class_id, cx, cy, w, h = lines[0].split()
    assert class_id == str(labelled.classes.index("cat"))
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.35, 0.45, 0.5, 0.5))
    assert "cat" in (out / "classes.txt").read_text()
    assert "names:" in (out / "data.yaml").read_text()


def img2label_paths(image_paths):
    """How ultralytics (v5/v8/v11) finds a label file for an image."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return [sb.join(p.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for p in image_paths]


def test_yolo_layout_is_what_the_trainer_resolves(labelled, tmp_path):
    """Every exported image must pair with a label under the trainer's own rule."""
    out = exporters.export_yolo(labelled, tmp_path / "yolo")
    images = sorted(str(p) for p in (out / "images" / "train").iterdir())
    assert images, "no images were placed in the dataset"
    for label_path in img2label_paths(images):
        assert Path(label_path).exists(), f"no label for {label_path}"


def test_yolo_data_yaml_points_at_the_export_not_the_source(labelled, tmp_path):
    out = exporters.export_yolo(labelled, tmp_path / "yolo")
    config = dict(
        line.split(": ", 1)
        for line in (out / "data.yaml").read_text().splitlines()
        if ": " in line and not line.startswith((" ", "#"))
    )
    assert config["path"] == str(out.resolve())
    assert config["train"] == "images/train"
    # With no split asked for, val has to point somewhere that exists.
    assert (out / config["val"]).is_dir()


def test_yolo_links_images_by_default_and_can_copy_or_skip(labelled, tmp_path):
    linked = exporters.export_yolo(labelled, tmp_path / "linked")
    assert (linked / "images" / "train" / "a.jpg").is_symlink()

    copied = exporters.export_yolo(labelled, tmp_path / "copied", images="copy")
    image = copied / "images" / "train" / "a.jpg"
    assert image.is_file() and not image.is_symlink()

    bare = exporters.export_yolo(labelled, tmp_path / "bare", images="none")
    assert list((bare / "images" / "train").iterdir()) == []
    assert (bare / "labels" / "train" / "a.txt").exists()


def split_of(out):
    return {
        split: sorted(p.name for p in (out / "labels" / split).iterdir())
        for split in ("train", "val", "test")
    }


def test_yolo_val_split_is_deterministic(project, tmp_path):
    for path in project.paths():
        project.set_annotations(path, [Annotation("cat", Box(0, 0, 0.5, 0.5))])

    first = split_of(exporters.export_yolo(project, tmp_path / "one", val_split=0.5))
    second = split_of(exporters.export_yolo(project, tmp_path / "two", val_split=0.5))
    assert first == second, "re-exporting must not reshuffle the split"
    assert sum(len(names) for names in first.values()) == 3


def test_yolo_splits_three_ways_in_roughly_the_right_proportions(image_dir, tmp_path):
    """Enough images that the hash-based split can be checked as a ratio."""
    from PIL import Image

    from auto_annotator.store import Project

    for index in range(200):
        Image.new("RGB", (20, 20)).save(image_dir / f"bulk_{index:03d}.jpg")
    project = Project(image_dir)
    for path in project.paths():
        project.set_annotations(path, [Annotation("cat", Box(0, 0, 0.5, 0.5))])

    out = exporters.export_yolo(project, tmp_path / "ds", val_split=0.2, test_split=0.1)
    counts = {split: len(names) for split, names in split_of(out).items()}
    total = sum(counts.values())
    assert total == 203
    assert 0.6 < counts["train"] / total < 0.8
    assert 0.12 < counts["val"] / total < 0.28
    assert 0.04 < counts["test"] / total < 0.18
    assert "test: images/test" in (out / "data.yaml").read_text()


def test_yolo_rejects_splits_that_leave_no_training_data(labelled, tmp_path):
    with pytest.raises(ValueError):
        exporters.export_yolo(labelled, tmp_path / "x", val_split=0.6, test_split=0.5)
    with pytest.raises(ValueError):
        exporters.export_yolo(labelled, tmp_path / "y", test_split=-0.1)


def test_yolo_flattens_nested_paths_without_collisions(project, tmp_path):
    project.set_annotations("nested/c.png", [Annotation("cat", Box(0, 0, 1, 1))])
    out = exporters.export_yolo(project, tmp_path / "yolo")
    assert (out / "labels" / "train" / "nested__c.txt").exists()
    assert (out / "images" / "train" / "nested__c.png").exists()


def test_yolo_rejects_nonsense_options(labelled, tmp_path):
    with pytest.raises(ValueError):
        exporters.export_yolo(labelled, tmp_path / "x", images="teleport")
    with pytest.raises(ValueError):
        exporters.export_yolo(labelled, tmp_path / "y", val_split=1.5)


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
