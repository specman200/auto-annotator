"""Command line entry point: ``auto-annotator <command>``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, exporters
from .engine import MERGE_KEEP_HUMAN, MERGE_MODES, Annotator
from .inference import BACKEND_HELP, load_detector
from .store import PROJECT_DIRNAME, Project


def invocation() -> str:
    """How this process was started, so printed hints can be pasted back in.

    The ``auto-annotator`` console script lands in a Scripts/bin directory that
    is often not on PATH (a plain ``pip install --user`` on Windows, or an
    unactivated virtualenv), and then ``python -m auto_annotator`` is what
    works — so tell people whichever one they are already using.
    """
    # Split on both separators rather than Path.name, which only understands
    # the host platform's — argv[0] is whatever launched us.
    name = (sys.argv[0] or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name.startswith("auto-annotator") or name.startswith("auto_annotator.exe"):
        return "auto-annotator"
    return f"{Path(sys.executable).stem} -m auto_annotator"


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
        prog=invocation(),
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
    export.add_argument("--test-split", type=float, default=0.0, metavar="F",
                        help="yolo: hold back this fraction of images for testing")
    export.add_argument("--image-mode", choices=("link", "copy", "none"), default="link",
                        help="yolo: symlink (default), copy, or omit the image files")

    merge = subparsers.add_parser(
        "merge",
        help="combine YOLO datasets (yours plus, say, a Roboflow zip) into one",
    )
    merge.add_argument(
        "sources", nargs="+", type=Path,
        help="datasets to merge: image folders annotated here, YOLO dataset "
             "directories, or .zip downloads",
    )
    merge.add_argument("-o", "--out", type=Path, required=True,
                       help="directory to write the merged dataset to")
    merge.add_argument("--val-split", type=float, default=0.2, metavar="F",
                       help="fraction for validation, for images with no split (default: 0.2)")
    merge.add_argument("--test-split", type=float, default=0.1, metavar="F",
                       help="fraction for testing, for images with no split (default: 0.1)")
    merge.add_argument("--resplit", action="store_true",
                       help="ignore the sources' own splits and re-split everything")
    merge.add_argument("--image-mode", choices=("copy", "link"), default="copy",
                       help="copy images (default) or symlink them")
    merge.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                       help="rename a class while merging, e.g. --map Car=car "
                            "(repeatable); this is how two datasets' names are unified")
    merge.add_argument("--drop-duplicates", action="store_true",
                       help="skip images that are byte-identical to one already merged")
    merge.add_argument("--dry-run", action="store_true",
                       help="report what would be merged without writing anything")

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
    print(f"review them with:  {invocation()} serve {args.images}")
    return 1 if result["errors"] else 0


def cmd_export(args: argparse.Namespace) -> int:
    project = Project(args.images)
    path = exporters.export(
        project, args.format, args.out, only_annotated=not args.include_empty,
        images=args.image_mode, val_split=args.val_split, test_split=args.test_split,
    )
    print(f"wrote {args.format} export to {path}")
    if args.format == "yolo":
        _report_splits(path, {"val": args.val_split, "test": args.test_split})
    return 0


def _report_splits(dataset_dir: Path, requested: dict) -> None:
    counts = {}
    for split in ("train", "val", "test"):
        labels = dataset_dir / "labels" / split
        counts[split] = len(list(labels.glob("*.txt"))) if labels.is_dir() else 0
    print("  " + ", ".join(f"{split}: {count}" for split, count in counts.items()))
    for split, fraction in requested.items():
        if fraction and not counts[split]:
            print(
                f"  ! asked for {fraction:.0%} {split} but no image landed there — "
                f"splits are assigned by hashing each filename, which is only "
                f"approximate on a small dataset",
                file=sys.stderr,
            )


def cmd_merge(args: argparse.Namespace) -> int:
    from . import datasets as ds

    class_map = {}
    for pair in args.map:
        if "=" not in pair:
            print(f"error: --map needs OLD=NEW, got {pair!r}", file=sys.stderr)
            return 2
        old, _, new = pair.partition("=")
        class_map[old] = new

    scratch = Path(args.out) / ".unpacked"
    loaded = []
    for source in args.sources:
        source = Path(source)
        if (source / PROJECT_DIRNAME).is_dir():
            # An image folder annotated here: export it first, then merge that.
            # The requested ratios apply to these images — a downloaded dataset
            # keeps the splits it shipped with unless --resplit says otherwise.
            project = Project(source)
            staged = scratch / f"{source.name}-export"
            exporters.export_yolo(
                project, staged, images="link",
                val_split=args.val_split, test_split=args.test_split,
            )
            print(f"exported {len(project.annotated_records())} annotated images "
                  f"from {source}")
            loaded.append(ds.load_dataset(staged, source=source.name))
        else:
            loaded.append(
                ds.load_dataset(source, source=source.stem,
                                extract_to=scratch / source.stem)
            )

    for dataset in loaded:
        counts = ", ".join(f"{k}: {v}" for k, v in dataset.counts().items())
        print(f"  {dataset.source:<24} {len(dataset.items)} images ({counts}), "
              f"{len(dataset.names)} classes [{dataset.layout} layout]")

    if args.dry_run:
        names, _, origin = ds.unify_names(loaded, class_map)
        print(f"\nwould merge into {len(names)} classes:")
        for index, name in enumerate(names):
            print(f"  {index:>3}  {name:<24} from {', '.join(origin[name])}")
        print("\n(dry run: nothing written)")
        return 0

    ratios = ds.normalize_ratios(None, args.val_split, args.test_split)
    report = ds.merge_datasets(
        loaded, args.out, ratios=ratios, resplit=args.resplit,
        image_mode=args.image_mode, class_map=class_map,
        drop_duplicates=args.drop_duplicates,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    print()
    print(report.describe())
    if not report.per_split.get("train"):
        print("\nerror: the merged dataset has no training images", file=sys.stderr)
        return 1
    print(f"\ntrain with:  yolo detect train data={Path(args.out).resolve()}/data.yaml")
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
    print(f"\nuse them as:  {invocation()} serve ./images --model onnx:yolov8n.onnx")
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
    "merge": cmd_merge,
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
