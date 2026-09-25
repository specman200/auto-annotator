"""Saving onto a drive where files get locked underneath us.

A sync client (Box, OneDrive, Dropbox) or a virus scanner holds a file open
for a moment while it uploads or reads it. On Windows the rename that makes
a save atomic then fails with "access is denied". These tests inject that
failure, since it cannot be produced on this platform.
"""

import json
import os

import pytest

from auto_annotator import store
from auto_annotator.schema import Annotation, Box


def windows_lock(times, real=os.replace):
    """A replacement for os.replace that fails the first ``times`` calls."""
    state = {"calls": 0}

    def flaky(src, dst, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] <= times:
            raise PermissionError(
                5, "Access is denied", str(src), None, str(dst)
            )
        return real(src, dst, *args, **kwargs)

    flaky.state = state
    return flaky


def test_a_brief_lock_is_retried_and_the_save_succeeds(project, monkeypatch, image_dir):
    flaky = windows_lock(3)
    monkeypatch.setattr(store.os, "replace", flaky)
    monkeypatch.setattr(store, "REPLACE_DELAYS", (0, 0, 0, 0, 0))

    project.set_annotations("a.jpg", [Annotation("cat", Box(0.1, 0.1, 0.5, 0.5))])

    assert flaky.state["calls"] == 4, "it should have retried, not given up"
    sidecar = image_dir / ".auto-annotator" / "annotations" / "a.jpg.json"
    assert json.loads(sidecar.read_text())["annotations"][0]["label"] == "cat"


def test_a_lock_that_never_clears_falls_back_to_writing_in_place(project, monkeypatch, image_dir, caplog):
    """Losing atomicity for one save beats losing the annotations."""
    monkeypatch.setattr(store.os, "replace", windows_lock(99))
    monkeypatch.setattr(store, "REPLACE_DELAYS", (0, 0))

    project.set_annotations("a.jpg", [Annotation("dog", Box(0, 0, 1, 1))])

    sidecar = image_dir / ".auto-annotator" / "annotations" / "a.jpg.json"
    assert json.loads(sidecar.read_text())["annotations"][0]["label"] == "dog"
    assert "wrote it in place" in caplog.text


def test_no_temporary_files_are_left_behind(project, monkeypatch, image_dir):
    monkeypatch.setattr(store.os, "replace", windows_lock(99))
    monkeypatch.setattr(store, "REPLACE_DELAYS", (0,))
    project.set_annotations("a.jpg", [Annotation("cat", Box(0, 0, 1, 1))])

    annotations = image_dir / ".auto-annotator" / "annotations"
    assert list(annotations.glob("*.tmp")) == [], "a synced folder would fill with these"


def test_an_unwritable_file_reports_what_to_do(project, monkeypatch):
    monkeypatch.setattr(store.os, "replace", windows_lock(99))
    monkeypatch.setattr(store, "REPLACE_DELAYS", (0,))

    def refuse(self, *args, **kwargs):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(store.Path, "write_text", refuse)

    with pytest.raises(OSError) as error:
        project.set_annotations("a.jpg", [Annotation("cat", Box(0, 0, 1, 1))])
    message = str(error.value)
    assert "cannot save annotations" in message
    assert "sync client" in message and "pausing the sync client" in message


def test_concurrent_saves_do_not_share_a_temporary_file(project, monkeypatch, image_dir):
    """Two saves at once used to write to the same .tmp and race."""
    seen = []
    real_replace = os.replace

    def record(src, dst, *args, **kwargs):
        seen.append(os.path.basename(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(store.os, "replace", record)
    for index in range(4):
        project.set_annotations("a.jpg", [Annotation(f"cat{index}", Box(0, 0, 1, 1))])

    assert len(set(seen)) == len(seen), f"temporary names repeated: {seen}"


def test_the_api_explains_a_failed_save_instead_of_a_500(image_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from auto_annotator.server import create_app

    monkeypatch.setattr(store.os, "replace", windows_lock(99))
    monkeypatch.setattr(store, "REPLACE_DELAYS", (0,))

    def refuse(self, *args, **kwargs):
        raise PermissionError(5, "Access is denied")

    app = create_app(image_dir, model_spec="mock")
    monkeypatch.setattr(store.Path, "write_text", refuse)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.put(
            "/api/images/a.jpg/annotations",
            json={"annotations": [{"label": "cat",
                                   "box": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}]},
        )
    assert response.status_code == 503
    assert "sync client" in response.json()["detail"]
