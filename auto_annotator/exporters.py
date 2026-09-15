"""Write a project's annotations out in the usual training-dataset formats."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from .schema import ImageRecord
from .store import Project

FORMATS = ("coco", "yolo", "voc", "csv")


def _records(project: Project, only_annotated: bool) -> List[ImageRecord]:
    records = project.records()
    return [r for r in records if r.annotations] if only_annotated else records


def _class_list(project: Project, records: List[ImageRecord]) -> List[str]:
    classes = list(project.classes)
    for record in records:
        for annotation in record.annotations:
            if annotation.label not in classes:
                classes.append(annotation.label)
    return classes


def export_coco(
    project: Project, out_path: Path, only_annotated: bool = True
) -> Path:
    """One COCO detection JSON, boxes as ``[x, y, width, height]`` in pixels."""
    records = _records(project, only_annotated)
    classes = _class_list(project, records)
    class_ids = {name: index + 1 for index, name in enumerate(classes)}

    images, annotations = [], []
    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record.path,
                "width": record.width,
                "height": record.height,
            }
        )
        for annotation in record.annotations:
            x1, y1, x2, y2 = annotation.box.to_pixels(record.width, record.height)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": class_ids[annotation.label],
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "area": round((x2 - x1) * (y2 - y1), 2),
                    "iscrowd": 0,
                    "score": annotation.score,
                    "source": annotation.source,
                }
            )

    payload = {
        "info": {
            "description": f"auto-annotator export of {project.image_root.name}",
            "date_created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name, "supercategory": "none"}
            for name, index in class_ids.items()
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def export_yolo(
    project: Project, out_dir: Path, only_annotated: bool = True
) -> Path:
    """Darknet/YOLO layout: one ``.txt`` per image plus ``classes.txt``/``data.yaml``."""
    records = _records(project, only_annotated)
    classes = _class_list(project, records)
    class_ids = {name: index for index, name in enumerate(classes)}

    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        lines = []
        for annotation in record.annotations:
            box = annotation.box
            cx = (box.x1 + box.x2) / 2
            cy = (box.y1 + box.y2) / 2
            lines.append(
                f"{class_ids[annotation.label]} "
                f"{cx:.6f} {cy:.6f} {box.width:.6f} {box.height:.6f}"
            )
        target = labels_dir / (record.path.replace("/", "__").rsplit(".", 1)[0] + ".txt")
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    (out_dir / "data.yaml").write_text(
        f"path: {project.image_root}\ntrain: .\nval: .\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return out_dir


def export_voc(
    project: Project, out_dir: Path, only_annotated: bool = True
) -> Path:
    """Pascal VOC: one XML per image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for record in _records(project, only_annotated):
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = str(project.image_root.name)
        ET.SubElement(root, "filename").text = record.name
        ET.SubElement(root, "path").text = str(project.abs_path(record.path))
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(record.width)
        ET.SubElement(size, "height").text = str(record.height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for annotation in record.annotations:
            x1, y1, x2, y2 = annotation.box.to_pixels(record.width, record.height)
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = annotation.label
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(round(x1)))
            ET.SubElement(bndbox, "ymin").text = str(int(round(y1)))
            ET.SubElement(bndbox, "xmax").text = str(int(round(x2)))
            ET.SubElement(bndbox, "ymax").text = str(int(round(y2)))

        target = out_dir / (record.path.replace("/", "__").rsplit(".", 1)[0] + ".xml")
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return out_dir


def export_csv(
    project: Project, out_path: Path, only_annotated: bool = True
) -> Path:
    """Flat CSV — one row per box, pixel coordinates."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "label", "xmin", "ymin", "xmax", "ymax",
             "image_width", "image_height", "score", "source"]
        )
        for record in _records(project, only_annotated):
            for annotation in record.annotations:
                x1, y1, x2, y2 = annotation.box.to_pixels(record.width, record.height)
                writer.writerow(
                    [
                        record.path,
                        annotation.label,
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        record.width,
                        record.height,
                        "" if annotation.score is None else round(annotation.score, 4),
                        annotation.source,
                    ]
                )
    return out_path


def export(
    project: Project,
    fmt: str,
    out: Optional[Path] = None,
    only_annotated: bool = True,
) -> Path:
    """Export in ``fmt``, choosing a sensible default destination when ``out`` is None."""
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; use one of {', '.join(FORMATS)}")
    default_root = project.project_dir / "exports"
    if fmt == "coco":
        return export_coco(project, out or default_root / "annotations.coco.json", only_annotated)
    if fmt == "csv":
        return export_csv(project, out or default_root / "annotations.csv", only_annotated)
    if fmt == "yolo":
        return export_yolo(project, out or default_root / "yolo", only_annotated)
    return export_voc(project, out or default_root / "voc", only_annotated)
