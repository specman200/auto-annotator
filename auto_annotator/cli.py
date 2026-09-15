"""Command line entry point: ``auto-annotator <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, exporters
from .engine import MERGE_KEEP_HUMAN, MERGE_MODES, Annotator
from .inference import BACKEND_HELP, load_detector
from .store import Project


def _classes(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    if Path(value).is_file():  # a file with one class per line
        return [
            line.strip()
            for line in Path(value).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [name.strip() for name in value.split(",") if name.strip()]


def _detector_options(args: argparse.Namespace) -> dict:
    """Collect the backend options the user actually specified."""
    options = {}
    if getattr(args, "labels", None):
        options["labels"] = _classes(args.labels)
    if getattr(args, "device", None):
        options["device"] = args.device
    return options


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-annotator",
        description="Auto-label images with a local model, then fix the labels in a GUI.",
    )
    parser.add_argument("--version", action="version", version=f"auto-annotator {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("images", type=Path, help="folder of images to annotate")
        sub.add_argument(
            "-m", "--model", default="mock",
            help="model spec, e.g. onnx:yolov8n.onnx, ultralytics:yolov8n.pt, mock",
        )
        sub.add_argument("-c", "--conf", type=float, default=0.25,
                         help="confidence threshold (default: 0.25)")
        sub.add_argument("--classes", help="comma-separated class names, or a file with one per line")
        sub.add_argument(
            "--labels",
            help="class names the MODEL outputs, in model order "
                 "(only needed for custom exports that carry no names)",
        )
        sub.add_argument(
            "--device",
            help="inference device for backends that offer a choice "
                 "(openvino: CPU, GPU, NPU, AUTO; torchvision/ultralytics: cpu, cuda)",
        )

    serve = subparsers.add_parser("serve", help="open the annotation GUI in a browser")
    add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--open", action="store_true", help="open a browser window")
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    annotate = subparsers.add_parser("annotate", help="run the model over a folder, no GUI")
    add_common(annotate)
    annotate.add_argument("--merge", choices=MERGE_MODES, default=MERGE_KEEP_HUMAN)
    annotate.add_argument("--all", action="store_true",
                          help="also re-run images already marked reviewed")
    annotate.add_argument("--only-new", action="store_true",
                          help="skip images that already have boxes")

    export = subparsers.add_parser("export", help="write annotations to a dataset format")
    export.add_argument("images", type=Path)
    export.add_argument("-f", "--format", choices=exporters.FORMATS, default="coco")
    export.add_argument("-o", "--out", type=Path, help="output file or directory")
    export.add_argument("--include-empty", action="store_true",
                        help="also export images that have no boxes")
    export.add_argument("--val-split", type=float, default=0.0, metavar="F",
                        help="yolo: hold back this fraction of images for validation")
    export.add_argument("--image-mode", choices=("link", "copy", "none"), default="link",
                        help="yolo: symlink (default), copy, or omit the image files")

    stats = subparsers.add_parser("stats", help="summarize a project")
    stats.add_argument("images", type=Path)

    subparsers.add_parser("backends", help="list the available inference backends")

    demo = subparsers.add_parser("demo", help="generate sample images and open the GUI")
    demo.add_argument("--dir", type=Path, default=Path("./demo-images"))
    demo.add_argument("--count", type=int, default=6)
    demo.add_argument("--port", type=int, default=8000)
    return parser


# --------------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    url = f"http://{args.host}:{args.port}"
    print(f"auto-annotator {__version__}")
    print(f"  images : {Path(args.images).resolve()}")
    print(f"  model  : {args.model}")
    print(f"  GUI    : {url}")

    if args.open:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    options = _detector_options(args)
    app = create_app(
        args.images, model_spec=args.model, conf=args.conf,
        classes=_classes(args.classes), **options,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    project = Project(args.images, classes=_classes(args.classes))
    project.save_project_file()
    options = _detector_options(args)
    annotator = Annotator(project, model_spec=args.model, conf=args.conf, **options)
    print(f"{len(project)} images in {project.image_root}")

    width = len(str(len(project)))

    def progress(done: int, total: int, current: str) -> None:
        print(f"  [{done:>{width}}/{total}] {current}", flush=True)

    result = annotator.annotate_all(
        merge=args.merge,
        skip_reviewed=not args.all,
        skip_annotated=args.only_new,
        progress=progress,
    )
    print(
        f"annotated {result['processed']} images "
        f"({result['boxes']} boxes, {result['skipped']} skipped)"
    )
    for error in result["errors"]:
        print(f"  ! {error['path']}: {error['error']}", file=sys.stderr)
    print(f"review them with:  auto-annotator serve {args.images}")
    return 1 if result["errors"] else 0


def cmd_export(args: argparse.Namespace) -> int:
    project = Project(args.images)
    path = exporters.export(
        project, args.format, args.out, only_annotated=not args.include_empty,
        images=args.image_mode, val_split=args.val_split,
    )
    print(f"wrote {args.format} export to {path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    project = Project(args.images)
    stats = project.stats()
    print(f"{project.image_root}")
    print(f"  images   : {stats['images']}")
    print(f"  boxes    : {stats['boxes']}")
    for status, count in stats["by_status"].items():
        print(f"  {status:<9}: {count}")
    if stats["per_class"]:
        print("  classes  :")
        for name, count in stats["per_class"].items():
            print(f"    {name:<20} {count}")
    return 0


def cmd_backends(_args: argparse.Namespace) -> int:
    print("available backends:")
    for name, help_text in BACKEND_HELP.items():
        installed = "" if _backend_installed(name) else "   (not installed)"
        print(f"  {name:<12} {help_text}{installed}")

    devices = _openvino_devices()
    if devices:
        print(f"\nopenvino devices on this machine: {', '.join(devices)}")
    print("\nuse them as:  auto-annotator serve ./images --model onnx:yolov8n.onnx")
    return 0


def _backend_installed(name: str) -> bool:
    import importlib.util

    module = {
        "onnx": "onnxruntime",
        "openvino": "openvino",
        "ultralytics": "ultralytics",
        "torchvision": "torchvision",
    }.get(name)
    if module is None:
        return True  # mock needs nothing
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _openvino_devices() -> List[str]:
    try:
        import openvino

        return list(openvino.Core().available_devices)
    except Exception:
        return []


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import make_demo_images

    target = Path(args.dir)
    made = make_demo_images(target, count=args.count)
    print(f"wrote {made} demo images to {target.resolve()}")
    demo_args = argparse.Namespace(
        images=target, model="mock", conf=0.25, classes="object,person,vehicle",
        labels=None, device=None, host="127.0.0.1", port=args.port,
        open=False, reload=False,
    )
    return cmd_serve(demo_args)


COMMANDS = {
    "serve": cmd_serve,
    "annotate": cmd_annotate,
    "export": cmd_export,
    "stats": cmd_stats,
    "backends": cmd_backends,
    "demo": cmd_demo,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
