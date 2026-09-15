import json

import pytest
from fastapi.testclient import TestClient

from auto_annotator.server import create_app


@pytest.fixture
def client(image_dir):
    app = create_app(image_dir, model_spec="mock", conf=0.2, classes=["cat"])
    with TestClient(app) as test_client:
        yield test_client


def test_project_endpoint_describes_everything_the_gui_needs(client, image_dir):
    data = client.get("/api/project").json()
    assert data["root"] == str(image_dir.resolve())
    assert data["classes"] == ["cat"]
    assert data["model"]["kind"] == "mock" and data["model"]["conf"] == 0.2
    assert [image["name"] for image in data["images"]] == ["a.jpg", "b.png", "c.png"]
    assert data["stats"]["images"] == 3


def test_image_detail_and_file_serving(client):
    detail = client.get("/api/images/nested/c.png").json()
    assert detail["width"] == 64 and detail["annotations"] == []
    response = client.get("/api/file/nested/c.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_unknown_image_is_a_404(client):
    assert client.get("/api/images/ghost.jpg").status_code == 404
    assert client.get("/api/file/ghost.jpg").status_code == 404


def test_saving_annotations_persists_them_and_registers_the_class(client, image_dir):
    payload = {
        "annotations": [
            {"label": "otter", "box": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.4},
             "source": "human", "score": None}
        ],
        "status": "reviewed",
    }
    data = client.put("/api/images/a.jpg/annotations", json=payload).json()
    assert data["image"]["status"] == "reviewed"
    assert "otter" in data["classes"]
    assert data["stats"]["boxes"] == 1

    sidecar = image_dir / ".auto-annotator" / "annotations" / "a.jpg.json"
    assert json.loads(sidecar.read_text())["annotations"][0]["label"] == "otter"


def test_malformed_annotations_are_rejected(client):
    response = client.put("/api/images/a.jpg/annotations", json={"annotations": [{"box": {}}]})
    assert response.status_code == 422


def test_status_endpoint_validates_the_value(client):
    assert client.post("/api/images/a.jpg/status", json={"status": "reviewed"}).status_code == 200
    assert client.post("/api/images/a.jpg/status", json={"status": "nope"}).status_code == 422


def test_predict_one_image(client):
    data = client.post("/api/images/a.jpg/predict", json={"conf": 0.1}).json()
    assert data["image"]["annotations"]
    assert data["image"]["status"] == "predicted"
    assert all(a["source"] == "model" for a in data["image"]["annotations"])


def test_predict_respects_the_merge_mode(client):
    client.put("/api/images/a.jpg/annotations", json={
        "annotations": [{"label": "mine", "box": {"x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1},
                         "source": "human"}]})
    kept = client.post("/api/images/a.jpg/predict", json={"merge": "keep-human"}).json()
    assert "mine" in [a["label"] for a in kept["image"]["annotations"]]
    replaced = client.post("/api/images/a.jpg/predict", json={"merge": "replace"}).json()
    assert "mine" not in [a["label"] for a in replaced["image"]["annotations"]]


def test_batch_job_runs_to_completion(client):
    job = client.post("/api/jobs/predict", json={"conf": 0.1}).json()
    for _ in range(200):
        status = client.get(f"/api/jobs/{job['id']}").json()
        if status["state"] != "running":
            break
    assert status["state"] == "done"
    assert status["result"]["processed"] == 3
    assert status["stats"]["boxes"] > 0


def test_unknown_job_is_a_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_classes_can_be_replaced(client):
    assert client.put("/api/classes", json={"classes": ["a", "b", "a"]}).json()["classes"] == ["a", "b"]


def test_model_can_be_reconfigured_at_runtime(client):
    data = client.put("/api/model", json={"conf": 0.6}).json()
    assert data["conf"] == 0.6
    assert client.put("/api/model", json={"spec": "nonsense:x"}).status_code == 422


def test_export_writes_the_requested_format(client, image_dir):
    client.post("/api/images/a.jpg/predict", json={"conf": 0.1})
    data = client.post("/api/export", json={"format": "yolo"}).json()
    assert (image_dir / ".auto-annotator" / "exports" / "yolo" / "classes.txt").exists()
    assert data["path"].endswith("yolo")
    assert client.post("/api/export", json={"format": "bogus"}).status_code == 422


def test_export_download_returns_a_file(client):
    client.post("/api/images/a.jpg/predict", json={"conf": 0.1})
    response = client.get("/api/export/download?fmt=coco")
    assert response.status_code == 200
    assert json.loads(response.content)["images"]
    assert client.get("/api/export/download?fmt=voc").status_code == 422


def test_rescan_picks_up_files_added_while_serving(client, image_dir):
    from PIL import Image

    Image.new("RGB", (20, 20)).save(image_dir / "late.jpg")
    data = client.post("/api/project/rescan").json()
    assert "late.jpg" in [image["path"] for image in data["images"]]


def test_the_gui_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Auto Annotator" in response.text
    assert client.get("/app.js").status_code == 200
