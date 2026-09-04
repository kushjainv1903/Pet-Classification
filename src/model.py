from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
from tensorflow.keras import layers


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    constructor: Callable[..., tf.keras.Model]
    preprocess_input: Callable[[tf.Tensor], tf.Tensor]


BACKBONES = {
    "efficientnetb0": BackboneConfig(
        name="efficientnetb0",
        constructor=tf.keras.applications.EfficientNetB0,
        preprocess_input=tf.keras.applications.efficientnet.preprocess_input,
    ),
    "mobilenetv2": BackboneConfig(
        name="mobilenetv2",
        constructor=tf.keras.applications.MobileNetV2,
        preprocess_input=tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "resnet50": BackboneConfig(
        name="resnet50",
        constructor=tf.keras.applications.ResNet50,
        preprocess_input=tf.keras.applications.resnet50.preprocess_input,
    ),
}


def get_backbone_config(backbone_name: str) -> BackboneConfig:
    backbone_key = backbone_name.lower()
    if backbone_key not in BACKBONES:
        available = ", ".join(sorted(BACKBONES))
        raise ValueError(f"Unknown backbone '{backbone_name}'. Choose one of: {available}")
    return BACKBONES[backbone_key]


@tf.keras.utils.register_keras_serializable(package="pet_classifier")
class BackbonePreprocessing(layers.Layer):
    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name.lower()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return get_backbone_config(self.backbone_name).preprocess_input(inputs)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"backbone_name": self.backbone_name})
        return config


def build_model(
    num_classes: int,
    img_size: int = 224,
    backbone_name: str = "efficientnetb0",
    dropout: float = 0.35,
) -> tf.keras.Model:
    backbone_config = get_backbone_config(backbone_name)

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.RandomFlip("horizontal", name="augment_flip")(inputs)
    x = layers.RandomRotation(0.08, name="augment_rotation")(x)
    x = layers.RandomZoom(0.15, name="augment_zoom")(x)
    x = layers.RandomContrast(0.15, name="augment_contrast")(x)

    x = BackbonePreprocessing(backbone_config.name, name=f"{backbone_config.name}_preprocess")(x)

    backbone = backbone_config.constructor(
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
    backbone_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    if len(backbone_layers) != 1:
        raise ValueError("Could not find a backbone layer to unfreeze")

    backbone = backbone_layers[0]
    backbone.trainable = True
    for layer in backbone.layers[:-trainable_layers]:
        layer.trainable = False
