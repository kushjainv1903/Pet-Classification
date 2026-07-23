from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.dataset import load_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the pet breed from one image.")
    parser.add_argument("--model", default="models/best_model.keras", help="Path to a trained Keras model.")
    parser.add_argument("--labels", default="models/labels.json", help="Path to labels.json.")
    parser.add_argument("--image", required=True, help="Path to a pet image.")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show.")
    return parser.parse_args()


def load_image(path: str | Path, img_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(str(path))
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (img_size, img_size), antialias=True)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)


def main() -> None:
    args = parse_args()
    class_names = load_labels(args.labels)
    model = tf.keras.models.load_model(args.model)
    image = load_image(args.image, args.img_size)

    probabilities = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][: args.top_k]

    print(f"Image: {args.image}")
    for rank, index in enumerate(top_indices, start=1):
        confidence = probabilities[index] * 100
        print(f"{rank}. {class_names[index]} - {confidence:.2f}%")


if __name__ == "__main__":
    main()
