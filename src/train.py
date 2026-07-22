import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import get_datasets
from model import build_model


MODEL_DIR = "../models"
OUTPUT_DIR = "../outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load datasets
# -----------------------------
train_ds, val_ds, test_ds = get_datasets()


# -----------------------------
# Build model
# -----------------------------
model = build_model()
model.summary()


# -----------------------------
# Callbacks (mandatory)
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        f"{MODEL_DIR}/best_stage1.keras",
        save_best_only=True
    )
]


# -----------------------------
# Stage 1: train classifier head
# -----------------------------
print("\nStage 1: Training head only...\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8,
    callbacks=callbacks
)


# -----------------------------
# Stage 2: fine-tuning
# -----------------------------
print("\nStage 2: Fine-tuning backbone...\n")

model.layers[1].trainable = True   # EfficientNet backbone

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)


# -----------------------------
# Save final model
# -----------------------------
model.save(f"{MODEL_DIR}/final_model.keras")


# -----------------------------
# Plot accuracy curves
# -----------------------------
plt.figure()


plt.plot(history1.history["accuracy"] + history2.history["accuracy"])
plt.plot(history1.history["val_accuracy"] + history2.history["val_accuracy"])

plt.legend(["train", "val"])
plt.title("Training Curve")
plt.savefig(f"{OUTPUT_DIR}/training_curve.png")

print("\nTraining complete. Model saved.")
