import json

import pytest

from auto_annotator.cli import main


def test_annotate_then_export_round_trip(image_dir, capsys):
    assert main(["annotate", str(image_dir), "--model", "mock", "--conf", "0.1"]) == 0
    output = capsys.readouterr().out
    assert "annotated 3 images" in output

    assert main(["export", str(image_dir), "-f", "coco"]) == 0
    export_path = image_dir / ".auto-annotator" / "exports" / "annotations.coco.json"
    assert len(json.loads(export_path.read_text())["annotations"]) > 0


def test_annotate_only_new_skips_labelled_images(image_dir, capsys):
    main(["annotate", str(image_dir), "--model", "mock"])
    capsys.readouterr()
    main(["annotate", str(image_dir), "--model", "mock", "--only-new"])
    assert "3 skipped" in capsys.readouterr().out


def test_classes_can_come_from_a_file(image_dir, tmp_path, capsys):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("cat\ndog\n\n")
    main(["annotate", str(image_dir), "--classes", str(classes_file)])
    stored = json.loads((image_dir / ".auto-annotator" / "project.json").read_text())
    assert stored["classes"][:2] == ["cat", "dog"]


def test_stats_command_reports_counts(image_dir, capsys):
    main(["annotate", str(image_dir), "--model", "mock"])
    capsys.readouterr()
    assert main(["stats", str(image_dir)]) == 0
    out = capsys.readouterr().out
    assert "images   : 3" in out and "predicted: 3" in out


def test_backends_command_lists_the_registry(capsys):
    assert main(["backends"]) == 0
    assert "onnx" in capsys.readouterr().out


def test_missing_folder_is_a_clean_error(tmp_path, capsys):
    assert main(["stats", str(tmp_path / "absent")]) == 2
    assert "error:" in capsys.readouterr().err


def test_export_rejects_an_unknown_format(image_dir):
    with pytest.raises(SystemExit):
        main(["export", str(image_dir), "-f", "parquet"])
