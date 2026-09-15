# Auto Annotator

Label an image folder with a model that runs **on your machine**, then fix what the
model got wrong in a browser GUI. Nothing is uploaded anywhere: inference, the
images and the annotations all stay local.

![The annotation GUI](docs/screenshot.png)

## Install

```bash
pip install -e .            # core: FastAPI, Pillow, numpy
pip install -e ".[onnx]"    # + onnxruntime, for YOLO .onnx models
pip install -e ".[dev]"     # + pytest, for the test suite
```

## Try it in 30 seconds

```bash
auto-annotator demo         # makes sample images and opens the GUI on :8000
```

That runs the `mock` backend, which invents deterministic boxes — enough to see
how review works without downloading weights.

## Real work

```bash
# 1. pre-label a folder (no GUI, good for large batches)
auto-annotator annotate ./images --model onnx:yolov8n.onnx --conf 0.3

# 2. correct the labels in the browser
auto-annotator serve ./images --model onnx:yolov8n.onnx --open

# 3. hand the dataset to a trainer
auto-annotator export ./images --format yolo -o ./dataset
```

`serve` alone is enough — the GUI can run the model per image or across the whole
folder — but pre-labelling first means every image already has boxes when you
start reviewing.

### Getting a model

| Backend | Install | Use |
| --- | --- | --- |
| `onnx` | `pip install onnxruntime` | `--model onnx:yolov8n.onnx` |
| `ultralytics` | `pip install ultralytics` | `--model ultralytics:yolov8n.pt` |
| `torchvision` | `pip install torch torchvision` | `--model torchvision:fasterrcnn_resnet50_fpn` |
| `mock` | — | `--model mock` |

`auto-annotator backends` prints the same list. A YOLO `.onnx` file comes from
ultralytics:

```bash
yolo export model=yolov8n.pt format=onnx     # writes yolov8n.onnx
```

The ONNX backend reads YOLOv5/v7 and YOLOv8/v11 exports, takes class names from
the file's metadata when they are there, and otherwise assumes COCO. For a custom
model, pass the names in model order:

```bash
auto-annotator serve ./images --model onnx:parts.onnx --labels bolt,nut,washer
```

If the name count does not match what the model predicts, boxes come back as
`class_0`, `class_1`, … rather than quietly wearing the wrong names.

## The GUI

* **Draw** — drag on empty space. **Move** — drag inside a box. **Resize** — drag a handle.
* **Relabel** — click a class pill, pick from the row's dropdown, or press `1`–`9`.
* **Run the model** — `Run on image` (`R`) for one, `Run on all…` for the folder,
  with a progress dialog you can cancel.
* **Confidence** and **merge mode** in the toolbar decide what a run does with
  boxes that are already there:
  * `keep my boxes` — replace old predictions, keep anything you drew or edited
  * `replace all` — start over from the model's output
  * `append` — add to what is there
* **Review flow** — `V` marks the image reviewed and jumps to the next one.
  Reviewed images are skipped by `Run on all…`, so a second pass only touches
  what you have not checked.

Boxes are dashed while they are the model's and solid once you have touched them;
the dots in the image list are grey (new), amber (predicted) and green (reviewed).

Everything saves itself half a second after you stop editing — `Ctrl+S` forces it,
`Ctrl+Z` / `Ctrl+Shift+Z` undo and redo, scroll zooms, space-drag pans, `F` fits.

## Where annotations live

Next to your images, in a sidecar folder — the images themselves are never
touched or moved:

```
images/
  cat.jpg
  .auto-annotator/
    project.json                # class list and settings
    annotations/cat.json        # one file per image
    exports/                    # whatever you export
```

Boxes are stored normalized (0–1), so resizing or re-encoding an image later does
not invalidate its labels. One file per image means an interrupted save can cost
at most one image's work.

## Export formats

`coco`, `yolo`, `voc` and `csv`:

```bash
auto-annotator export ./images -f coco          # .auto-annotator/exports/annotations.coco.json
auto-annotator export ./images -f yolo -o ./ds  # labels/*.txt + classes.txt + data.yaml
auto-annotator export ./images -f csv --include-empty
```

Exports carry the score and whether each box came from the model or a human, so
you can audit or filter later.

## Commands

```
auto-annotator serve    <images> [--model SPEC] [--conf F] [--classes ...] [--labels ...] [--port N] [--open]
auto-annotator annotate <images> [--model SPEC] [--conf F] [--merge MODE] [--all] [--only-new]
auto-annotator export   <images> [-f coco|yolo|voc|csv] [-o PATH] [--include-empty]
auto-annotator stats    <images>
auto-annotator backends
auto-annotator demo     [--dir D] [--count N]
```

## Adding a backend

Subclass `Detector`, return normalized boxes, register the class:

```python
# my_backend.py
from auto_annotator.inference.base import Detector
from auto_annotator.schema import Box, Detection

class MyDetector(Detector):
    kind = "mine"

    def load(self):
        self.model = ...            # called once, lazily
        self._loaded = True

    def _predict(self, image_path, conf):
        return [Detection("widget", Box(0.1, 0.1, 0.4, 0.4), 0.9)]
```

```python
from auto_annotator.inference import BACKENDS
BACKENDS["mine"] = "my_backend:MyDetector"
```

Anything with a `predict(path, conf)` returning `Detection`s also works directly:

```python
from auto_annotator.engine import Annotator
from auto_annotator.store import Project

project = Project("./images")
Annotator(project, detector=MyDetector()).annotate_all()
```

## HTTP API

The GUI is a client of a small JSON API on the same port, so scripts can drive it too:

| Method | Path | Does |
| --- | --- | --- |
| `GET` | `/api/project` | classes, stats, model, image list |
| `GET` | `/api/images/{path}` | one image with its annotations |
| `PUT` | `/api/images/{path}/annotations` | replace an image's annotations |
| `POST` | `/api/images/{path}/predict` | run the model on one image |
| `POST` | `/api/jobs/predict` | start a folder-wide run (poll `/api/jobs/{id}`) |
| `PUT` | `/api/model` | swap model or threshold while serving |
| `POST` | `/api/export` | write an export |

## Tests

```bash
pytest
```

The suite covers the store, merge policies, exporters and the HTTP API, and runs
the ONNX backend through a real `onnxruntime` session built on the fly, so the
letterboxing and coordinate maths are checked without downloading weights.
