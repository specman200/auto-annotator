import json

import pytest

from auto_annotator.schema import Annotation, Box
from auto_annotator.store import Project


def test_scan_finds_images_recursively_and_ignores_other_files(project):
    assert project.paths() == ["a.jpg", "b.png", "nested/c.png"]


def test_sizes_are_not_read_during_the_scan(project):
    """Opening every file to measure it makes startup crawl on a synced drive."""
    assert (project.get("a.jpg").width, project.get("a.jpg").height) == (0, 0)


def test_sizes_are_read_on_demand_and_remembered(project, monkeypatch):
    record = project.ensure_size("a.jpg")
    assert (record.width, record.height) == (200, 100)

    def fail(path):
        raise AssertionError("the file should not be opened twice")

    monkeypatch.setattr("auto_annotator.store.image_size", fail)
    assert project.ensure_size("a.jpg").width == 200


def test_ensure_sizes_fills_a_whole_project_with_progress(project):
    seen = []
    project.ensure_sizes(progress=lambda done, total: seen.append((done, total)))
    assert [r.width for r in project.records()] == [200, 120, 64]
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_the_walk_skips_the_project_folder_and_hidden_directories(project, image_dir):
    from PIL import Image

    (image_dir / ".cache").mkdir()
    Image.new("RGB", (10, 10)).save(image_dir / ".cache" / "thumb.jpg")
    assert list(project.walk_images()) == ["a.jpg", "b.png", "nested/c.png"]


def test_a_directory_link_loop_does_not_hang_the_walk(project, image_dir):
    """Windows user folders are full of junctions that point at an ancestor."""
    try:
        (image_dir / "nested" / "loop").symlink_to(image_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not make the link")
    assert sorted(project.walk_images()) == ["a.jpg", "b.png", "nested/c.png"]


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
    project = Project(image_dir)
    assert project.get("a.jpg").annotations == []
    assert project.ensure_size("a.jpg").width == 200


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


def test_a_slow_filesystem_does_not_delay_startup(image_dir, monkeypatch):
    """The scan used to open every image, so a cloud folder took forever.

    Startup must not depend on how long reading a file takes — that is what
    left people staring at "scanning" while their sync client downloaded the
    whole dataset.
    """
    import time

    opened = []

    def slow_read(path):
        opened.append(path)
        time.sleep(0.2)          # a cloud placeholder being hydrated
        return (640, 480)

    monkeypatch.setattr("auto_annotator.store.image_size", slow_read)

    started = time.monotonic()
    project = Project(image_dir)
    elapsed = time.monotonic() - started

    assert len(project) == 3
    assert opened == [], "the scan should not open image files at all"
    assert elapsed < 0.2, f"scanning took {elapsed:.2f}s, so it read the files"

    # ...and the size still arrives when something actually needs it
    assert project.ensure_size("a.jpg").width == 640
    assert len(opened) == 1


def test_only_the_images_being_exported_are_opened(project, tmp_path, monkeypatch):
    from auto_annotator import exporters
    from auto_annotator.schema import Annotation as _Annotation

    opened = []
    monkeypatch.setattr(
        "auto_annotator.store.image_size",
        lambda path: (opened.append(path), (100, 50))[1],
    )
    project.set_annotations("a.jpg", [_Annotation("cat", Box(0, 0, 1, 1))])
    exporters.export_coco(project, tmp_path / "out.json")

    assert len(opened) == 1, "an unannotated image should not be opened"
