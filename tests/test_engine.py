import time

import pytest

from auto_annotator.engine import (
    MERGE_APPEND,
    MERGE_KEEP_HUMAN,
    MERGE_REPLACE,
    Annotator,
    JobRunner,
)
from auto_annotator.inference.base import Detector
from auto_annotator.schema import Annotation, Box, Detection


class TwoBoxDetector(Detector):
    """Always predicts the same two boxes, one of them low-scoring."""

    kind = "fake"

    def _predict(self, image_path, conf):
        return [
            Detection("cat", Box(0.1, 0.1, 0.3, 0.3), 0.9),
            Detection("dog", Box(0.5, 0.5, 0.7, 0.7), 0.2),
        ]


class ExplodingDetector(Detector):
    kind = "boom"

    def _predict(self, image_path, conf):
        raise RuntimeError("model blew up")


@pytest.fixture
def annotator(project):
    return Annotator(project, detector=TwoBoxDetector(), conf=0.25)


def test_conf_threshold_drops_weak_detections(annotator):
    assert [a.label for a in annotator.annotate_image("a.jpg")] == ["cat"]
    assert len(annotator.annotate_image("b.png", conf=0.1, merge=MERGE_REPLACE)) == 2


def test_keep_human_merge_preserves_hand_drawn_boxes(project, annotator):
    project.set_annotations("a.jpg", [
        Annotation("mine", Box(0, 0, 0.1, 0.1), source="human"),
        Annotation("stale", Box(0, 0, 0.2, 0.2), source="model"),
    ])
    merged = annotator.annotate_image("a.jpg", merge=MERGE_KEEP_HUMAN)
    assert [a.label for a in merged] == ["mine", "cat"]


def test_replace_merge_drops_everything_first(project, annotator):
    project.set_annotations("a.jpg", [Annotation("mine", Box(0, 0, 0.1, 0.1))])
    assert [a.label for a in annotator.annotate_image("a.jpg", merge=MERGE_REPLACE)] == ["cat"]


def test_append_merge_keeps_old_model_boxes(project, annotator):
    annotator.annotate_image("a.jpg")
    merged = annotator.annotate_image("a.jpg", merge=MERGE_APPEND)
    assert [a.label for a in merged] == ["cat", "cat"]


def test_unknown_merge_mode_is_rejected(annotator):
    with pytest.raises(ValueError):
        annotator.annotate_image("a.jpg", merge="nonsense")


def test_predictions_mark_the_image_predicted(project, annotator):
    annotator.annotate_image("a.jpg")
    assert project.get("a.jpg").status == "predicted"


def test_reviewed_images_keep_their_status_and_are_skipped(project, annotator):
    project.set_status("a.jpg", "reviewed")
    result = annotator.annotate_all()
    assert result["skipped"] == 1 and result["processed"] == 2
    assert project.get("a.jpg").status == "reviewed"


def test_skip_annotated_leaves_existing_boxes_alone(project, annotator):
    project.set_annotations("b.png", [Annotation("mine", Box(0, 0, 0.1, 0.1))])
    result = annotator.annotate_all(skip_annotated=True)
    assert result["skipped"] == 1 and result["processed"] == 2


def test_batch_reports_progress_for_every_image(project, annotator):
    seen = []
    annotator.annotate_all(progress=lambda done, total, path: seen.append((done, path)))
    assert [s[0] for s in seen] == [1, 2, 3]


def test_a_failing_image_does_not_abort_the_batch(project):
    annotator = Annotator(project, detector=ExplodingDetector())
    result = annotator.annotate_all()
    assert result["processed"] == 0
    assert len(result["errors"]) == 3
    assert "blew up" in result["errors"][0]["error"]


def _wait_for(job_runner, job_id, states=("done", "error", "cancelled"), timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = job_runner.status(job_id)
        if status and status["state"] in states:
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job_runner.status(job_id)}")


def test_job_runner_runs_in_the_background_and_reports_a_result(project, annotator):
    runner = JobRunner()
    job = runner.start(annotator)
    finished = _wait_for(runner, job["id"])
    assert finished["state"] == "done"
    assert finished["result"]["processed"] == 3
    assert project.stats()["boxes"] == 3


def test_job_runner_reports_failures(project):
    runner = JobRunner()
    job = runner.start(Annotator(project, detector=ExplodingDetector()), paths=["a.jpg"])
    finished = _wait_for(runner, job["id"])
    # Per-image errors are collected, so the job itself still completes.
    assert finished["state"] == "done"
    assert finished["result"]["errors"]


def test_unknown_job_id_has_no_status():
    assert JobRunner().status("nope") is None
