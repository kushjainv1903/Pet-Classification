from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_datasets, load_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pet breed classifier.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", help="Oxford-IIIT Pet dataset directory.")
    parser.add_argument("--model", default="models/best_model.keras", help="Path to a trained Keras model.")
    parser.add_argument("--labels", default="models/labels.json", help="Path to labels.json.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--output-dir", default="output", help="Directory for reports and plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_labels(args.labels)
    bundle = get_datasets(data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    model = tf.keras.models.load_model(str(args.model))

    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in bundle.test:
        probabilities = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
