"""HTTP API + static GUI.

The browser talks to this over plain JSON; everything runs on localhost and
the images never leave the machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, exporters
from .engine import MERGE_KEEP_HUMAN, MERGE_MODES, Annotator, JobRunner
from .inference import BACKEND_HELP, load_detector
from .schema import STATUS_REVIEWED, Annotation
from .store import Project

WEB_DIR = Path(__file__).parent / "web"


class NoCacheStatic(StaticFiles):
    """Serve the GUI with revalidation forced on every load.

    The page, its stylesheet and its script are one unit: a browser holding a
    cached copy of one and a fresh copy of another renders a broken layout that
    looks like a bug in the tool. ETags still make the revalidation a cheap 304,
    and this is localhost either way.
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


class AppState:
    """Everything the request handlers share."""

    def __init__(self, project: Project, annotator: Annotator):
        self.project = project
        self.annotator = annotator
        self.jobs = JobRunner()


def create_app(
    image_root: Path,
    model_spec: str = "mock",
    conf: float = 0.25,
    classes: Optional[List[str]] = None,
    **detector_options: Any,
) -> FastAPI:
    project = Project(image_root, classes=classes)
    project.save_project_file()
    annotator = Annotator(
        project, model_spec=model_spec, conf=conf, **detector_options
    )
    state = AppState(project, annotator)

    app = FastAPI(title="auto-annotator", version=__version__)
    app.state.annotator_state = state

    # ------------------------------------------------------------- project

    @app.get("/api/project")
    def get_project() -> Dict[str, Any]:
        return {
            "root": str(state.project.image_root),
            "version": __version__,
            "classes": state.project.classes,
            "stats": state.project.stats(),
            "model": {
                **state.annotator.detector.describe(),
                "conf": state.annotator.conf,
            },
            "backends": BACKEND_HELP,
            "merge_modes": list(MERGE_MODES),
            "images": [record.summary() for record in state.project.records()],
        }

    @app.post("/api/project/rescan")
    def rescan() -> Dict[str, Any]:
        state.project.rescan()
        return {
            "images": [record.summary() for record in state.project.records()],
            "stats": state.project.stats(),
        }

    @app.put("/api/classes")
    def put_classes(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        classes = state.project.set_classes(payload.get("classes") or [])
        return {"classes": classes}

    # -------------------------------------------------------------- images

    @app.get("/api/images")
    def list_images(status: Optional[str] = None) -> Dict[str, Any]:
        records = state.project.records()
        if status:
            records = [r for r in records if r.status == status]
        return {"images": [record.summary() for record in records]}

    @app.get("/api/images/{path:path}")
    def get_image(path: str) -> Dict[str, Any]:
        return _record(path).to_dict()

    @app.put("/api/images/{path:path}/annotations")
    def put_annotations(path: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        _record(path)  # 404 before we parse anything
        try:
            annotations = [
                Annotation.from_dict(item) for item in payload.get("annotations", [])
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"bad annotation: {exc}")
        record = state.project.set_annotations(
            path, annotations, status=payload.get("status")
        )
        return {"image": record.to_dict(), "stats": state.project.stats(),
                "classes": state.project.classes}

    @app.post("/api/images/{path:path}/status")
    def post_status(path: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        _record(path)
        try:
            record = state.project.set_status(path, payload.get("status", STATUS_REVIEWED))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"image": record.summary(), "stats": state.project.stats()}

    @app.get("/api/file/{path:path}")
    def get_file(path: str) -> FileResponse:
        _record(path)
        return FileResponse(state.project.abs_path(path))

    # ----------------------------------------------------------- inference

    @app.post("/api/images/{path:path}/predict")
    def predict_one(path: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        _record(path)
        try:
            state.annotator.annotate_image(
                path,
                conf=payload.get("conf"),
                merge=payload.get("merge", MERGE_KEEP_HUMAN),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
        return {
            "image": state.project.get(path).to_dict(),
            "stats": state.project.stats(),
            "classes": state.project.classes,
        }

    @app.post("/api/jobs/predict")
    def start_job(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        job = state.jobs.start(
            state.annotator,
            paths=payload.get("paths"),
            conf=payload.get("conf"),
            merge=payload.get("merge", MERGE_KEEP_HUMAN),
            skip_reviewed=bool(payload.get("skip_reviewed", True)),
            skip_annotated=bool(payload.get("skip_annotated", False)),
        )
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        job = state.jobs.status(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        if job["state"] in ("done", "error", "cancelled"):
            job["stats"] = state.project.stats()
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Dict[str, Any]:
        return {"cancelled": state.jobs.cancel(job_id)}

    @app.put("/api/model")
    def set_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        spec = payload.get("spec")
        if spec:
            try:
                state.annotator.detector = load_detector(spec)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc))
        if payload.get("conf") is not None:
            state.annotator.conf = float(payload["conf"])
        return {**state.annotator.detector.describe(), "conf": state.annotator.conf}

    # -------------------------------------------------------------- export

    @app.post("/api/export")
    def post_export(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        fmt = payload.get("format", "coco")
        out = payload.get("out")
        try:
            path = exporters.export(
                state.project,
                fmt,
                Path(out) if out else None,
                only_annotated=bool(payload.get("only_annotated", True)),
                images=payload.get("images", "link"),
                val_split=float(payload.get("val_split", 0.0)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"format": fmt, "path": str(path)}

    @app.get("/api/export/download")
    def download_export(fmt: str = Query("coco")) -> FileResponse:
        if fmt not in ("coco", "csv"):
            raise HTTPException(
                status_code=422, detail="only coco and csv are single-file downloads"
            )
        path = exporters.export(state.project, fmt)
        return FileResponse(path, filename=Path(path).name)

    # ------------------------------------------------------------- helpers

    def _record(path: str):
        try:
            return state.project.get(path)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such image: {path}")

    @app.exception_handler(ValueError)
    def value_error_handler(_request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    if WEB_DIR.is_dir():
        app.mount("/", NoCacheStatic(directory=WEB_DIR, html=True), name="web")

    return app
