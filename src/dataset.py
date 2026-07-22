import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from datasets import load_dataset

IMG_SIZE = 224
BATCH_SIZE = 32


def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


def hf_to_tf(hf_ds):

    def generator():
        for item in hf_ds:
            img = item["image"].convert("RGB")   # force 3 channels
            img = np.array(img).astype("float32")
            label = item["labels"]
            yield img, label

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((None, None, 3), tf.float32),
            tf.TensorSpec((), tf.int64)
        )
    )

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


def get_datasets():

    ds = load_dataset("microsoft/cats_vs_dogs")

    split = ds["train"].train_test_split(test_size=0.3, seed=42)
    temp = split["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = hf_to_tf(split["train"])
    val_ds   = hf_to_tf(temp["train"])
    test_ds  = hf_to_tf(temp["test"])

    return train_ds, val_ds, test_ds


# -------- DEBUG --------
if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_datasets()

    print("Counting batches manually...")

    train_batches = sum(1 for _ in train_ds)
    val_batches   = sum(1 for _ in val_ds)
    test_batches  = sum(1 for _ in test_ds)

    print("Train batches:", train_batches)
    print("Val batches:", val_batches)
    print("Test batches:", test_batches)


    for images, labels in train_ds.take(1):
        print("Shape:", images.shape)
        print("Labels:", labels[:10])
