"""Reading, splitting and merging YOLO datasets."""

import zipfile
from collections import Counter

import pytest
from PIL import Image

from auto_annotator import datasets as ds


def write_label(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_image(path, colour=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), colour).save(path)


@pytest.fixture
def roboflow_dir(tmp_path):
    """The layout a Roboflow YOLOv8/v11 export actually ships."""
    root = tmp_path / "roboflow"
    for offset, (split, count) in enumerate((("train", 4), ("valid", 2), ("test", 1))):
        for index in range(count):
            stem = f"{split}_{index}_jpg.rf.hash{index}"
            # A distinct colour per image: identical bytes would trip the
            # duplicate detector and make the counts below meaningless.
            make_image(root / split / "images" / f"{stem}.jpg",
                       (index * 9, 40 + offset * 11, 60))
            write_label(root / split / "labels" / f"{stem}.txt",
                        ["0 0.5 0.5 0.2 0.2", "1 0.25 0.25 0.1 0.1"])
    (root / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
        "nc: 2\nnames: ['person', 'car']\n"
    )
    (root / "README.roboflow.txt").write_text("exported by roboflow")
    return root


@pytest.fixture
def ours_dir(tmp_path):
    """The layout this tool exports: images/<split>, labels/<split>."""
    root = tmp_path / "mine"
    for index in range(4):
        make_image(root / "images" / "train" / f"mine_{index}.png", (200, index * 9, 30))
        write_label(root / "labels" / "train" / f"mine_{index}.txt", ["0 0.4 0.4 0.2 0.2"])
    (root / "data.yaml").write_text("names:\n  0: car\n  1: truck\n")
    return root


def zip_up(directory, archive):
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in directory.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(directory))
    return archive


# ------------------------------------------------------------------- reading


def test_reads_a_roboflow_layout_with_its_splits(roboflow_dir):
    dataset = ds.load_dataset(roboflow_dir)
    assert dataset.layout == "roboflow"
    assert dataset.names == ["person", "car"]
    assert dataset.counts() == {"train": 4, "val": 2, "test": 1}   # "valid" -> "val"


def test_reads_our_own_export_layout(ours_dir):
    dataset = ds.load_dataset(ours_dir)
    assert dataset.layout == "split-last"
    assert dataset.names == ["car", "truck"]
    assert dataset.counts() == {"train": 4}


def test_reads_a_zip_and_looks_through_a_wrapper_directory(roboflow_dir, tmp_path):
    nested = tmp_path / "wrapped"
    (nested / "my-dataset-v3").mkdir(parents=True)
    for path in roboflow_dir.iterdir():
        target = nested / "my-dataset-v3" / path.name
        if path.is_dir():
            target.mkdir()
            for child in path.rglob("*"):
                if child.is_file():
                    destination = target / child.relative_to(path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(child.read_bytes())
        else:
            target.write_bytes(path.read_bytes())

    archive = zip_up(nested, tmp_path / "dataset.zip")
    dataset = ds.load_dataset(archive, extract_to=tmp_path / "unpacked")
    assert dataset.names == ["person", "car"]
    assert sum(dataset.counts().values()) == 7


def test_a_zip_cannot_write_outside_the_target(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "nope")
    with pytest.raises(ValueError, match="outside the target"):
        ds.extract_zip(archive, tmp_path / "out")


def test_a_non_yolo_download_says_so(tmp_path):
    coco = tmp_path / "coco-export"
    coco.mkdir()
    (coco / "_annotations.coco.json").write_text("{}")
    with pytest.raises(ValueError, match="YOLO"):
        ds.load_dataset(coco)


def test_names_come_from_classes_txt_when_there_is_no_yaml(ours_dir):
    (ours_dir / "data.yaml").unlink()
    (ours_dir / "classes.txt").write_text("car\ntruck\n")
    assert ds.load_dataset(ours_dir).names == ["car", "truck"]


def test_missing_names_fall_back_to_the_label_ids(ours_dir):
    (ours_dir / "data.yaml").unlink()
    write_label(ours_dir / "labels" / "train" / "mine_0.txt", ["2 0.4 0.4 0.2 0.2"])
    dataset = ds.load_dataset(ours_dir)
    assert dataset.names == ["class_0", "class_1", "class_2"]
    assert any("no data.yaml" in warning for warning in dataset.warnings)


@pytest.mark.parametrize("body", [
    "names: ['a', 'b']",
    "names:\n  - a\n  - b",
    "names:\n  0: a\n  1: b",
])
def test_the_yaml_shapes_data_yaml_files_use(tmp_path, body):
    path = tmp_path / "data.yaml"
    path.write_text(f"train: ../train/images\n{body}\nnc: 2\n")
    assert ds._names_from_yaml(path) == ["a", "b"]


# ------------------------------------------------------------------ splitting


def test_split_assignment_is_deterministic_and_roughly_proportional():
    keys = [f"image_{index}.jpg" for index in range(3000)]
    first = [ds.assign_split(key, (0.7, 0.2, 0.1)) for key in keys]
    second = [ds.assign_split(key, (0.7, 0.2, 0.1)) for key in keys]
    assert first == second

    counts = Counter(first)
    assert 0.65 < counts["train"] / len(keys) < 0.75
    assert 0.16 < counts["val"] / len(keys) < 0.24
    assert 0.06 < counts["test"] / len(keys) < 0.14


def test_adding_images_does_not_move_the_existing_ones():
    """Hashing names, rather than shuffling, keeps a split stable as data grows."""
    before = {key: ds.assign_split(key, (0.7, 0.2, 0.1)) for key in
              (f"image_{i}.jpg" for i in range(100))}
    after = {key: ds.assign_split(key, (0.7, 0.2, 0.1)) for key in
             (f"image_{i}.jpg" for i in range(400))}
    assert all(after[key] == split for key, split in before.items())


def test_ratios_must_leave_room_to_train_on():
    assert ds.normalize_ratios(None, 0.2, 0.1) == pytest.approx((0.7, 0.2, 0.1))
    with pytest.raises(ValueError):
        ds.normalize_ratios(None, 0.7, 0.4)
    with pytest.raises(ValueError):
        ds.normalize_ratios(None, -0.1, 0.1)


# -------------------------------------------------------------------- merging


def test_merging_rewrites_class_ids_onto_a_shared_list(ours_dir, roboflow_dir, tmp_path):
    """The whole point: 0 means "car" in one dataset and "person" in the other."""
    mine = ds.load_dataset(ours_dir, source="mine")
    theirs = ds.load_dataset(roboflow_dir, source="roboflow")
    report = ds.merge_datasets([mine, theirs], tmp_path / "merged")

    assert report.names == ["car", "truck", "person"]

    def named_rows(dataset):
        """Every box as (class name, geometry) — comparable across layouts."""
        found = Counter()
        for item in dataset.items:
            for class_id, rest in ds.read_label(item.label):
                found[(dataset.names[class_id], rest)] += 1
        return found

    expected = named_rows(mine) + named_rows(theirs)
    merged = named_rows(ds.load_dataset(tmp_path / "merged"))
    assert merged == expected, "a box changed class or geometry in the merge"
    assert sum(merged.values()) == 4 + 7 * 2


def test_merging_keeps_each_source_split_unless_told_otherwise(ours_dir, roboflow_dir, tmp_path):
    """A downloaded dataset's validation images are held out for a reason."""
    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    report = ds.merge_datasets(sources, tmp_path / "kept")
    assert report.per_source["roboflow"] == {"train": 4, "val": 2, "test": 1}

    resplit = ds.merge_datasets(sources, tmp_path / "resplit", resplit=True,
                                ratios=(0.6, 0.2, 0.2))
    assert sum(resplit.per_split.values()) == 11
    assert resplit.per_source["roboflow"] != {"train": 4, "val": 2, "test": 1}


def test_class_map_unifies_names_that_differ_only_in_spelling(ours_dir, roboflow_dir, tmp_path):
    (roboflow_dir / "data.yaml").write_text("names: ['Person', 'Car']\n")
    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    report = ds.merge_datasets(sources, tmp_path / "mapped",
                               class_map={"Car": "car", "Person": "person"})
    assert report.names == ["car", "truck", "person"]
    assert set(report.class_origin["car"]) == {"mine", "roboflow"}


def test_colliding_filenames_are_kept_apart(ours_dir, roboflow_dir, tmp_path):
    make_image(roboflow_dir / "train" / "images" / "mine_0.png", (1, 2, 3))
    write_label(roboflow_dir / "train" / "labels" / "mine_0.txt", ["0 0.5 0.5 0.2 0.2"])
    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    report = ds.merge_datasets(sources, tmp_path / "merged")

    assert report.renamed == 1
    images = [p.name for p in (tmp_path / "merged" / "images" / "train").iterdir()]
    assert "mine_0.png" in images and "roboflow_mine_0.png" in images


def test_identical_images_are_reported_and_can_be_dropped(ours_dir, roboflow_dir, tmp_path):
    """The same picture in train and val quietly inflates every metric."""
    duplicate = roboflow_dir / "train" / "images" / "copy_of_mine.png"
    duplicate.write_bytes((ours_dir / "images" / "train" / "mine_0.png").read_bytes())
    write_label(roboflow_dir / "train" / "labels" / "copy_of_mine.txt", ["0 .5 .5 .2 .2"])

    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    warned = ds.merge_datasets(sources, tmp_path / "warned")
    assert warned.duplicates and warned.dropped == 0
    assert "byte-identical" in warned.describe()

    dropped = ds.merge_datasets(sources, tmp_path / "dropped", drop_duplicates=True)
    assert dropped.dropped == 1
    written = sum(len(list((tmp_path / "dropped" / "images" / s).iterdir()))
                  for s in ds.SPLITS)
    assert written == len(sources[0].items) + len(sources[1].items) - 1


def test_segmentation_labels_survive_a_merge(ours_dir, roboflow_dir, tmp_path):
    """Polygon rows must keep their geometry; only the class id is rewritten."""
    polygon = "1 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5"
    write_label(roboflow_dir / "train" / "labels" / "train_0_jpg.rf.hash0.txt", [polygon])
    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    report = ds.merge_datasets(sources, tmp_path / "seg")

    merged = [
        line
        for label in (tmp_path / "seg" / "labels").rglob("*.txt")
        for line in label.read_text().splitlines()
        if len(line.split()) == 9
    ]
    assert merged, "the polygon row vanished"
    class_id, *coordinates = merged[0].split()
    assert report.names[int(class_id)] == "car"        # was id 1 in roboflow
    assert coordinates == polygon.split()[1:]


def test_the_merged_dataset_is_laid_out_the_way_a_trainer_reads_it(ours_dir, roboflow_dir, tmp_path):
    import os

    sources = [ds.load_dataset(ours_dir, source="mine"),
               ds.load_dataset(roboflow_dir, source="roboflow")]
    out = ds.merge_datasets(sources, tmp_path / "merged").out_dir

    config = (out / "data.yaml").read_text()
    assert f"path: {out.resolve()}" in config
    assert "train: images/train" in config and "val: images/val" in config

    for split in ds.SPLITS:
        for image in (out / "images" / split).iterdir():
            label = str(image).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
            label = label.rsplit(".", 1)[0] + ".txt"
            assert os.path.exists(label), f"no label for {image}"


def test_merging_warns_when_nothing_is_held_out(ours_dir, tmp_path):
    report = ds.merge_datasets([ds.load_dataset(ours_dir, source="mine")], tmp_path / "novalid")
    assert any("no validation images" in warning for warning in report.warnings)


def test_an_untrainable_merge_says_so(ours_dir, tmp_path):
    """Dropping every duplicate can leave nothing to train on; that must be loud."""
    for index in range(1, 4):   # make all of our images byte-identical
        target = ours_dir / "images" / "train" / f"mine_{index}.png"
        target.write_bytes((ours_dir / "images" / "train" / "mine_0.png").read_bytes())

    source = ds.load_dataset(ours_dir, source="mine")
    report = ds.merge_datasets([source], tmp_path / "thin", ratios=(0.0, 1.0, 0.0),
                               resplit=True, drop_duplicates=True)
    assert report.dropped == 3
    assert any("NO TRAINING IMAGES" in warning for warning in report.warnings)
    assert "NO TRAINING IMAGES" in report.describe()
