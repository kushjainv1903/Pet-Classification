from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from PIL import Image, UnidentifiedImageError

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_DATA_DIR = Path("data")
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SEED = 42
EXPECTED_CLASS_COUNT = 132
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LABELS_PATH = Path("models") / "labels.json"


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    class_name: str
    species: str
    source: str


@dataclass(frozen=True)
class ClassMappingEntry:
    index: int
    class_name: str
    species: str
    source: str


@dataclass(frozen=True)
class DatasetSplit:
    image_paths: list[str]
    labels: list[int]
    records: list[ImageRecord]

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
    class_mapping: tuple[ClassMappingEntry, ...]
    skipped_images: int = 0
    train_records: tuple[ImageRecord, ...] = ()
    val_records: tuple[ImageRecord, ...] = ()
    test_records: tuple[ImageRecord, ...] = ()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def normalize_breed_name(name: str) -> str:
    name = re.sub(r"^n\d+-", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("._")


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def is_tensorflow_decodable(path: Path) -> bool:
    """Validate that TensorFlow can decode the image file.

    This uses TensorFlow's actual image decoder to ensure images that pass
    through build_manifest() will not cause decoding errors during training.
    """
    try:
        image_bytes = tf.io.read_file(str(path))
        decoded = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        # Force evaluation to catch any decoding errors
        _ = decoded.shape
        return True
    except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
        return False


def discover_class_directories(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[tuple[str, str, Path]]:
    root = Path(data_dir)
    dog_root = root / "stanford-dog" / "Images"
    cat_root = root / "oxford-cat"

    if not dog_root.exists():
        raise FileNotFoundError(f"Stanford Dogs image folder not found: {dog_root}")
    if not cat_root.exists():
        raise FileNotFoundError(f"Oxford cat image folder not found: {cat_root}")

    class_dirs: list[tuple[str, str, Path]] = []
    for path in sorted(dog_root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_dir():
            class_dirs.append(("dog", normalize_breed_name(path.name), path))
    for path in sorted(cat_root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_dir():
            class_dirs.append(("cat", normalize_breed_name(path.name), path))

    return class_dirs


def build_manifest(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    validate_images: bool = True,
    expected_classes: int = EXPECTED_CLASS_COUNT,
) -> tuple[list[ImageRecord], list[str], tuple[ClassMappingEntry, ...], int]:
    class_dirs = discover_class_directories(data_dir)
    class_names = [class_name for _species, class_name, _path in class_dirs]

    duplicate_names = [name for name, count in Counter(class_names).items() if count > 1]
    if duplicate_names:
        names = ", ".join(sorted(duplicate_names))
        raise ValueError(f"Duplicate class names after normalization: {names}")

    if len(class_names) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, discovered {len(class_names)}")

    class_mapping = tuple(
        ClassMappingEntry(
            index=index,
            class_name=class_name,
            species=species,
            source=class_dir.name,
        )
        for index, (species, class_name, class_dir) in enumerate(class_dirs)
    )

    records: list[ImageRecord] = []
    skipped_images = 0
    for label, (species, class_name, class_dir) in enumerate(class_dirs):
        image_paths = sorted(
            (path for path in class_dir.iterdir() if is_supported_image(path)),
            key=lambda item: item.name.lower(),
        )
        for image_path in image_paths:
            if validate_images and not is_readable_image(image_path):
                skipped_images += 1
                continue
            if validate_images and not is_tensorflow_decodable(image_path):
                skipped_images += 1
                continue
            records.append(
                ImageRecord(
                    path=str(image_path),
                    label=label,
                    class_name=class_name,
                    species=species,
                    source=class_dir.name,
                )
            )

    missing = sorted(set(range(len(class_names))) - {record.label for record in records})
    if missing:
        names = ", ".join(class_names[index] for index in missing)
        raise ValueError(f"Classes without readable images: {names}")

    return records, class_names, class_mapping, skipped_images


def split_records(
    records: Iterable[ImageRecord],
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("train/val/test fractions must sum to 1.0")

    buckets: dict[int, list[ImageRecord]] = {}
    for record in records:
        buckets.setdefault(record.label, []).append(record)

    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []

    for label in sorted(buckets):
        items = list(buckets[label])
        random.Random(seed + label).shuffle(items)
        total = len(items)
        train_count = max(1, round(total * train_fraction))
        val_count = max(1, round(total * val_fraction))
        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)
        test_count = total - train_count - val_count
        if test_count < 1:
            test_count = 1
            train_count = total - val_count - test_count

        train_records.extend(items[:train_count])
        val_records.extend(items[train_count : train_count + val_count])
        test_records.extend(items[train_count + val_count :])

    return (
        _records_to_split(train_records),
        _records_to_split(val_records),
        _records_to_split(test_records),
    )


def _records_to_split(records: list[ImageRecord]) -> DatasetSplit:
    image_paths = [record.path for record in records]
    labels = [record.label for record in records]
    return DatasetSplit(image_paths=image_paths, labels=labels, records=records)


def validate_no_data_leakage(*splits: DatasetSplit) -> bool:
    split_paths = [set(split.image_paths) for split in splits]
    for left_index, left_paths in enumerate(split_paths):
        for right_paths in split_paths[left_index + 1 :]:
            if left_paths & right_paths:
                return False
    return True


def decode_image(image_path: tf.Tensor, label: tf.Tensor, img_size: int) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
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
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    validate_images: bool = True,
    labels_path: str | Path | None = DEFAULT_LABELS_PATH,
) -> DatasetBundle:
    records, class_names, class_mapping, skipped_images = build_manifest(data_dir, validate_images=validate_images)
    if labels_path is not None:
        save_label_mapping(class_mapping, labels_path)

    train_split, val_split, test_split = split_records(
        records,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if not validate_no_data_leakage(train_split, val_split, test_split):
        raise ValueError("Data leakage detected: at least one image appears in multiple splits")

    return DatasetBundle(
        train=make_tf_dataset(train_split, img_size, batch_size, shuffle=True, seed=seed),
        val=make_tf_dataset(val_split, img_size, batch_size, shuffle=False, seed=seed),
        test=make_tf_dataset(test_split, img_size, batch_size, shuffle=False, seed=seed),
        class_names=class_names,
        train_count=len(train_split),
        val_count=len(val_split),
        test_count=len(test_split),
        class_mapping=class_mapping,
        skipped_images=skipped_images,
        train_records=tuple(train_split.records),
        val_records=tuple(val_split.records),
        test_records=tuple(test_split.records),
    )


def save_labels(class_names: list[str], path: str | Path) -> None:
    labels_path = Path(path)
    if labels_path.exists():
        existing_mapping = load_label_mapping(labels_path)
        existing_class_names = [entry.class_name for entry in existing_mapping]
        has_species = all(entry.species != "unknown" for entry in existing_mapping)
        if existing_class_names == class_names and has_species:
            return

    class_mapping = tuple(
        ClassMappingEntry(index=index, class_name=class_name, species="unknown", source=class_name)
        for index, class_name in enumerate(class_names)
    )
    save_label_mapping(class_mapping, labels_path)


def save_label_mapping(class_mapping: Iterable[ClassMappingEntry], path: str | Path) -> None:
    labels_path = Path(path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "index": entry.index,
            "class_name": entry.class_name,
            "species": entry.species,
            "source": entry.source,
        }
        for entry in class_mapping
    ]
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_label_mapping(path: str | Path) -> list[ClassMappingEntry]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("labels.json must contain a list")  # noqa: TRY004

    entries: list[ClassMappingEntry] = []
    for item in payload:
        if isinstance(item, str):
            entries.append(
                ClassMappingEntry(
                    index=len(entries),
                    class_name=item,
                    species="unknown",
                    source=item,
                )
            )
            continue
        entries.append(
            ClassMappingEntry(
                index=int(item["index"]),
                class_name=str(item["class_name"]),
                species=str(item["species"]),
                source=str(item.get("source", item["class_name"])),
            )
        )

    expected_indices = list(range(len(entries)))
    actual_indices = [entry.index for entry in entries]
    if actual_indices != expected_indices:
        raise ValueError("labels.json indices must be sequential and ordered from 0")

    return entries


def load_labels(path: str | Path) -> list[str]:
    return [entry.class_name for entry in load_label_mapping(path)]


def summarize_split(split: DatasetSplit, class_names: list[str]) -> dict[str, int]:
    counts = Counter(split.labels)
    return {class_names[label]: counts[label] for label in sorted(counts)}


if __name__ == "__main__":
    bundle = get_datasets()
    total = bundle.train_count + bundle.val_count + bundle.test_count
    train_split = DatasetSplit(
        [record.path for record in bundle.train_records],
        [record.label for record in bundle.train_records],
        list(bundle.train_records),
    )
    val_split = DatasetSplit(
        [record.path for record in bundle.val_records],
        [record.label for record in bundle.val_records],
        list(bundle.val_records),
    )
    test_split = DatasetSplit(
        [record.path for record in bundle.test_records],
        [record.label for record in bundle.test_records],
        list(bundle.test_records),
    )
    no_leakage = validate_no_data_leakage(train_split, val_split, test_split)

    print(f"Classes discovered: {bundle.num_classes}")
    print(f"Images used: {total}")
    print(f"Skipped unreadable images: {bundle.skipped_images}")
    print(f"Train: {bundle.train_count}")
    print(f"Validation: {bundle.val_count}")
    print(f"Test: {bundle.test_count}")
    print(f"No data leakage: {no_leakage}")
    print("Saved labels: models/labels.json")
    print("First 10 classes:")
    for class_name in bundle.class_names[:10]:
        print(f"- {class_name}")
