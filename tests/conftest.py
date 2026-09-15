import pytest
from PIL import Image

from auto_annotator.store import Project


@pytest.fixture
def image_dir(tmp_path):
    """Three images, one of them in a subfolder."""
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (200, 100), "red").save(tmp_path / "a.jpg")
    Image.new("RGB", (120, 90), "green").save(tmp_path / "b.png")
    Image.new("RGB", (64, 64), "blue").save(tmp_path / "nested" / "c.png")
    (tmp_path / "notes.txt").write_text("not an image")
    return tmp_path


@pytest.fixture
def project(image_dir):
    return Project(image_dir, classes=["cat", "dog"])
