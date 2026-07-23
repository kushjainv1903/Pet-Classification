# Pet Breed Classification

Train and use a CNN image classifier for the 37 breeds in the Oxford-IIIT Pet dataset. The project is designed for practical pet photos, including images captured from mobile phones.

## What It Does

- Reads the official Oxford-IIIT Pet annotations from `annotations/list.txt`, `trainval.txt`, and `test.txt`.
- Trains a 37-class transfer-learning CNN with EfficientNetB0 or MobileNetV2.
- Saves the best model, label map, training curves, confusion matrix, and classification report.
- Predicts the most likely breed from a single uploaded or phone-captured image.

## Project Layout

```text
pet_classifier/
├── data/
│   └── oxford-iiit-pet/
│       ├── annotations/
│       └── images/
├── models/
├── output/
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

`data/`, `models/`, and `output/` are local-only folders and should not be pushed to GitHub.

## Setup

TensorFlow does not currently support Python 3.14. Use Python 3.11 or 3.12 for the virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the dataset here:

```text
data/oxford-iiit-pet/
├── annotations/
│   ├── list.txt
│   ├── trainval.txt
│   └── test.txt
└── images/
    ├── Abyssinian_1.jpg
    └── ...
```

## Train

```powershell
python -m src.train --data-dir data/oxford-iiit-pet --epochs 20 --batch-size 32
```

Useful options:

```powershell
python -m src.train --backbone mobilenetv2 --epochs 15 --fine-tune-epochs 3
```

Training writes:

- `models/best_model.keras`
- `models/final_model.keras`
- `models/labels.json`
- `output/training_curves.png`

## Evaluate

```powershell
python -m src.evaluate --data-dir data/oxford-iiit-pet --model models/best_model.keras
```

Evaluation writes:

- `output/confusion_matrix.png`
- `output/classification_report.txt`

## Predict One Image

```powershell
python -m src.predict --model models/best_model.keras --image path\to\mobile_photo.jpg
```

Example output:

```text
Image: phone_pet_photo.jpg
1. Maine Coon - 84.21%
2. Persian - 7.94%
3. Ragdoll - 4.11%
```

## Notes

The official dataset split is used for final testing. The official `trainval.txt` split is divided into train and validation sets in a class-balanced way during training.
