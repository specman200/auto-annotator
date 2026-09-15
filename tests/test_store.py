import json

import pytest

from auto_annotator.schema import Annotation, Box
from auto_annotator.store import Project


def test_scan_finds_images_recursively_and_ignores_other_files(project):
    assert project.paths() == ["a.jpg", "b.png", "nested/c.png"]


def test_image_sizes_are_read_from_the_files(project):
    record = project.get("a.jpg")
    assert (record.width, record.height) == (200, 100)


def test_annotations_survive_a_reload(project, image_dir):
    project.set_annotations(
        "a.jpg", [Annotation("dog", Box(0.1, 0.1, 0.4, 0.4), 0.7, "model")],
        status="predicted",
    )
    reopened = Project(image_dir)
    record = reopened.get("a.jpg")
    assert record.status == "predicted"
    assert record.annotations[0].label == "dog"
    assert record.annotations[0].box.x2 == 0.4


def test_new_labels_are_registered_as_classes(project, image_dir):
    project.set_annotations("b.png", [Annotation("horse", Box(0, 0, 1, 1))])
    assert "horse" in project.classes
    assert "horse" in Project(image_dir).classes


def test_corrupt_sidecar_does_not_break_loading(project, image_dir):
    project.set_annotations("a.jpg", [Annotation("dog", Box(0, 0, 1, 1))])
    sidecar = next((image_dir / ".auto-annotator" / "annotations").glob("*.json"))
    sidecar.write_text("{not json")
    record = Project(image_dir).get("a.jpg")
    assert record.annotations == [] and record.width == 200


def test_paths_outside_the_project_are_refused(project):
    with pytest.raises(ValueError):
        project.abs_path("../../etc/passwd")


def test_rescan_picks_up_new_files_and_drops_deleted_ones(project, image_dir):
    from PIL import Image

    Image.new("RGB", (10, 10)).save(image_dir / "d.jpg")
    (image_dir / "b.png").unlink()
    project.rescan()
    assert project.paths() == ["a.jpg", "d.jpg", "nested/c.png"]


def test_unknown_image_raises(project):
    with pytest.raises(KeyError):
        project.get("nope.jpg")


def test_set_classes_deduplicates_and_persists(project, image_dir):
    assert project.set_classes(["a", "b", "a", " "]) == ["a", "b"]
    assert json.loads(project.config_path.read_text())["classes"] == ["a", "b"]


def test_stats(project):
    project.set_annotations(
        "a.jpg",
        [Annotation("cat", Box(0, 0, 0.5, 0.5)), Annotation("cat", Box(0.5, 0.5, 1, 1))],
        status="reviewed",
    )
    stats = project.stats()
    assert stats == {
        "images": 3,
        "boxes": 2,
        "by_status": {"new": 2, "predicted": 0, "reviewed": 1},
        "per_class": {"cat": 2},
    }
