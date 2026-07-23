from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

from src.dataset import get_datasets, save_labels
from src.model import build_model, compile_model, unfreeze_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Oxford-IIIT Pet breed classifier.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", help="Oxford-IIIT Pet dataset directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Fine-tuning epochs after head training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=224, help="Square image size.")
    parser.add_argument("--backbone", default="efficientnetb0", choices=["efficientnetb0", "mobilenetv2"])
    parser.add_argument("--model-dir", default="models", help="Directory for saved models and labels.")
    parser.add_argument("--output-dir", default="output", help="Directory for plots and reports.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def plot_history(histories: list[tf.keras.callbacks.History], output_path: Path) -> None:
    accuracy: list[float] = []
    val_accuracy: list[float] = []
    loss: list[float] = []
    val_loss: list[float] = []

    for history in histories:
        accuracy.extend(history.history.get("accuracy", []))
        val_accuracy.extend(history.history.get("val_accuracy", []))
        loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label="train")
    plt.plot(val_accuracy, label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = get_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    save_labels(bundle.class_names, model_dir / "labels.json")

    print(f"Loaded {len(bundle.class_names)} breeds")
    print(f"Train: {bundle.train_count} | Val: {bundle.val_count} | Test: {bundle.test_count}")

    model = build_model(
        num_classes=len(bundle.class_names),
        img_size=args.img_size,
        backbone_name=args.backbone,
    )
    model.summary()

    head_epochs = max(1, args.epochs - args.fine_tune_epochs)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(model_dir / "best_model.keras"), save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3, verbose=1),
    ]

    histories: list[tf.keras.callbacks.History] = []
    print("\nStage 1: training classifier head\n")
    histories.append(
        model.fit(
            bundle.train,
            validation_data=bundle.val,
            epochs=head_epochs,
            callbacks=callbacks,
        )
    )

    if args.fine_tune_epochs > 0:
        print("\nStage 2: fine-tuning top backbone layers\n")
        unfreeze_backbone(model)
        compile_model(model, learning_rate=1e-5)
        histories.append(
            model.fit(
                bundle.train,
                validation_data=bundle.val,
                epochs=args.fine_tune_epochs,
                callbacks=callbacks,
            )
        )

    model.save(str(model_dir / "final_model.keras"))
    plot_history(histories, output_dir / "training_curves.png")
    print(f"\nTraining complete. Best model: {model_dir / 'best_model.keras'}")


if __name__ == "__main__":
    main()
