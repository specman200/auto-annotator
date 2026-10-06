import json
import sys

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


def test_invocation_matches_how_the_process_was_started(monkeypatch):
    """Printed hints must be pasteable in the shell the user is actually in."""
    from auto_annotator.cli import invocation

    monkeypatch.setattr(sys, "argv", ["C:\\py\\Scripts\\auto-annotator.exe", "serve"])
    assert invocation() == "auto-annotator"

    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/auto-annotator", "serve"])
    assert invocation() == "auto-annotator"

    # `python -m auto_annotator` puts __main__.py in argv[0].
    monkeypatch.setattr(sys, "argv", ["/app/auto_annotator/__main__.py", "serve"])
    assert invocation().endswith("-m auto_annotator")


def test_hints_point_at_the_module_form_when_run_as_a_module(image_dir, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["/app/auto_annotator/__main__.py", "annotate"])
    main(["annotate", str(image_dir), "--model", "mock"])
    assert "-m auto_annotator serve" in capsys.readouterr().out


def test_a_busy_port_is_reported_before_any_url(image_dir, capsys):
    """Printing a URL and then failing to bind sends people to a dead link."""
    import socket

    from auto_annotator.cli import port_is_free

    with socket.socket() as blocker:
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]

        assert port_is_free("127.0.0.1", port) is False
        assert main(["serve", str(image_dir), "--port", str(port)]) == 3

    output = capsys.readouterr()
    assert "already in use" in output.err
    assert f"--port {port + 1}" in output.err or "Try:" not in output.err
    assert "http://" not in output.out, "no URL should be offered when it cannot bind"


def test_a_free_port_is_recognised():
    import socket

    from auto_annotator.cli import port_is_free

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert port_is_free("127.0.0.1", port) is True


def test_the_url_is_only_printed_once_the_port_answers(image_dir):
    """The scan runs before the bind, so an early URL just refuses connections."""
    import socket
    import subprocess
    import sys
    import threading
    import time

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    process = subprocess.Popen(
        [sys.executable, "-m", "auto_annotator", "serve", str(image_dir), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    printed_at = []

    def watch():
        for line in process.stdout:
            if "ready" in line and "http://" in line:
                printed_at.append(time.monotonic())
                return

    threading.Thread(target=watch, daemon=True).start()
    try:
        accepted_at = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.settimeout(0.3)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    accepted_at = time.monotonic()
                    break
            time.sleep(0.02)

        assert accepted_at, "the server never started"
        deadline = time.monotonic() + 5
        while not printed_at and time.monotonic() < deadline:
            time.sleep(0.05)
        assert printed_at, "the ready URL was never printed"
        assert printed_at[0] >= accepted_at - 0.1, "the URL was printed before the port answered"
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_serve_reports_missing_annotations_when_a_shared_folder_lost_them(image_dir, capsys, monkeypatch):
    """Annotations live in a hidden folder, which copying and zipping skip."""
    import auto_annotator.cli as cli

    # stop before the server actually runs; we only want the startup report
    monkeypatch.setattr(cli, "_wait_until_serving", lambda *a, **k: False)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    main(["serve", str(image_dir), "--port", "0"])

    output = capsys.readouterr()
    assert "found 3 images" in output.out
    assert "hidden" in output.err and ".auto-annotator" in output.err


def test_serve_reports_how_much_was_annotated(image_dir, capsys, monkeypatch):
    import auto_annotator.cli as cli
    from auto_annotator.schema import Annotation, Box
    from auto_annotator.store import Project

    project = Project(image_dir)
    project.set_annotations("a.jpg", [Annotation("cat", Box(0, 0, 0.5, 0.5))])

    monkeypatch.setattr(cli, "_wait_until_serving", lambda *a, **k: False)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    main(["serve", str(image_dir), "--port", "0"])

    out = capsys.readouterr().out
    assert "1 boxes on 1 of them" in out


@pytest.fixture
def parent_with_annotated_subfolder(tmp_path):
    """Images annotated in a sub-folder, as "PPE Dataset/Arun" was."""
    from PIL import Image

    from auto_annotator.schema import Annotation, Box
    from auto_annotator.store import Project

    parent = tmp_path / "PPE Dataset"
    child = parent / "Arun"
    child.mkdir(parents=True)
    for index in range(4):
        Image.new("RGB", (640, 480)).save(child / f"shot_{index}.jpg")

    project = Project(child, classes=["helmet"])
    for path in project.paths():
        project.set_annotations(path, [Annotation("helmet", Box(0.2, 0.2, 0.5, 0.5))])
    return parent, child


def test_a_sub_folder_annotated_on_its_own_is_detected(parent_with_annotated_subfolder):
    from auto_annotator.store import Project

    parent, _ = parent_with_annotated_subfolder
    outer = Project(parent)
    assert len(outer) == 4, "the images are visible from the parent"
    assert outer.annotated_records() == [], "but their annotations are not"
    assert outer.nested_projects() == ["Arun"]


def test_exporting_the_parent_explains_why_it_is_empty(parent_with_annotated_subfolder, tmp_path, capsys):
    """This printed a split-ratio warning before, which pointed nowhere useful."""
    parent, _ = parent_with_annotated_subfolder
    assert main(["export", str(parent), "-f", "yolo", "-o", str(tmp_path / "out")]) == 1

    error = capsys.readouterr().err
    assert "nothing was exported" in error
    assert "Arun" in error, "it should name the folder that has the annotations"
    assert "hashing" not in error, "the split ratios are not the problem here"


def test_exporting_the_annotated_sub_folder_works(parent_with_annotated_subfolder, tmp_path, capsys):
    _, child = parent_with_annotated_subfolder
    out = tmp_path / "dataset"
    assert main(["export", str(child), "-f", "yolo", "-o", str(out)]) == 0

    assert "4 of 4 images" in capsys.readouterr().out
    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == 4
    assert "helmet" in (out / "classes.txt").read_text()


def test_an_empty_export_without_sub_projects_suggests_what_to_do(image_dir, tmp_path, capsys):
    assert main(["export", str(image_dir), "-f", "coco", "-o", str(tmp_path / "x.json")]) == 1
    error = capsys.readouterr().err
    assert "nothing was exported" in error
    assert "--include-empty" in error


def test_stats_flags_an_annotated_sub_folder(parent_with_annotated_subfolder, capsys):
    parent, _ = parent_with_annotated_subfolder
    assert main(["stats", str(parent)]) == 0
    assert "Arun" in capsys.readouterr().err
