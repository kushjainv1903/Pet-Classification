from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_DATA_DIR = Path("data/oxford-iiit-pet")
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 42


@dataclass(frozen=True)
class DatasetSplit:
    image_paths: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class DatasetBundle:
    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    train_count: int
    val_count: int
    test_count: int


def normalize_breed_name(name: str) -> str:
    return name.replace("_", " ").title()


def read_annotation_file(path: Path) -> list[tuple[str, int, int, int]]:
    rows: list[tuple[str, int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, class_id, species_id, breed_id = line.split()
            rows.append((image_id, int(class_id), int(species_id), int(breed_id)))
    return rows


def load_class_names(data_dir: Path) -> list[str]:
    rows = read_annotation_file(data_dir / "annotations" / "list.txt")
    by_class_id: dict[int, str] = {}
    for image_id, class_id, _species_id, _breed_id in rows:
        breed = image_id.rsplit("_", 1)[0]
        by_class_id[class_id - 1] = normalize_breed_name(breed)
    return [by_class_id[index] for index in sorted(by_class_id)]


def load_split(data_dir: Path, split_name: str) -> DatasetSplit:
    annotation_path = data_dir / "annotations" / f"{split_name}.txt"
    image_dir = data_dir / "images"
    image_paths: list[str] = []
    labels: list[int] = []

    for image_id, class_id, _species_id, _breed_id in read_annotation_file(annotation_path):
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by annotations: {image_path}")
        image_paths.append(str(image_path))
        labels.append(class_id - 1)

    return DatasetSplit(image_paths=image_paths, labels=labels)


def split_train_val(
    split: DatasetSplit,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[DatasetSplit, DatasetSplit]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    buckets: dict[int, list[tuple[str, int]]] = {}
    for image_path, label in zip(split.image_paths, split.labels):
        buckets.setdefault(label, []).append((image_path, label))

    train_items: list[tuple[str, int]] = []
    val_items: list[tuple[str, int]] = []

    for label in sorted(buckets):
        shuffled = list(buckets[label])
        random.Random(seed + label).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_items.extend(shuffled[:val_count])
        train_items.extend(shuffled[val_count:])

    return _items_to_split(train_items), _items_to_split(val_items)


def _items_to_split(items: Iterable[tuple[str, int]]) -> DatasetSplit:
    image_paths: list[str] = []
    labels: list[int] = []
    for image_path, label in items:
        image_paths.append(image_path)
        labels.append(label)
    return DatasetSplit(image_paths=image_paths, labels=labels)


def decode_image(image_path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (img_size, img_size), antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def make_tf_dataset(
    split: DatasetSplit,
    img_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((split.image_paths, split.labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(split), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: decode_image(path, label, img_size),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def get_datasets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    data_path = Path(data_dir)
    class_names = load_class_names(data_path)
    trainval_split = load_split(data_path, "trainval")
    test_split = load_split(data_path, "test")
    train_split, val_split = split_train_val(trainval_split, val_fraction=val_fraction, seed=seed)

    return DatasetBundle(
        train=make_tf_dataset(train_split, img_size, batch_size, shuffle=True, seed=seed),
        val=make_tf_dataset(val_split, img_size, batch_size, shuffle=False, seed=seed),
        test=make_tf_dataset(test_split, img_size, batch_size, shuffle=False, seed=seed),
        class_names=class_names,
        train_count=len(train_split),
        val_count=len(val_split),
        test_count=len(test_split),
    )


def save_labels(class_names: list[str], path: str | Path) -> None:
    labels_path = Path(path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_labels(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    bundle = get_datasets()
    print(f"Classes: {len(bundle.class_names)}")
    print(f"Train: {bundle.train_count} | Val: {bundle.val_count} | Test: {bundle.test_count}")
    print(bundle.class_names)
