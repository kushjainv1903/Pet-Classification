from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

BACKBONES = {
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
    "mobilenetv2": tf.keras.applications.MobileNetV2,
}


def build_model(
    num_classes: int,
    img_size: int = 224,
    backbone_name: str = "efficientnetb0",
    dropout: float = 0.35,
) -> tf.keras.Model:
    backbone_key = backbone_name.lower()
    if backbone_key not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose one of: {available}")

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.RandomFlip("horizontal", name="augment_flip")(inputs)
    x = layers.RandomRotation(0.08, name="augment_rotation")(x)
    x = layers.RandomZoom(0.15, name="augment_zoom")(x)
    x = layers.RandomContrast(0.15, name="augment_contrast")(x)

    backbone = BACKBONES[backbone_key](
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="breed")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pet_breed_classifier")
    compile_model(model, learning_rate=1e-3)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_backbone(model: tf.keras.Model, trainable_layers: int = 30) -> None:
    backbone = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.lower().startswith(("efficientnet", "mobilenet"))
    )
    backbone.trainable = True
    for layer in backbone.layers[:-trainable_layers]:
        layer.trainable = False
