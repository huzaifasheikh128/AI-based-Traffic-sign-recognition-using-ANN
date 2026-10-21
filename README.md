# AI-Based Traffic Sign Recognition Using Artificial Neural Networks

University project: a multilayer perceptron (MLP) that classifies 43 German traffic signs from the GTSRB dataset.

Stage 1 in this repository is the machine learning pipeline only. Flask and Gradio are not included yet.

## Project structure

- `ml/dataset_loader.py` — finds and loads the GTSRB dataset
- `ml/preprocess.py` — resize to 32×32 and normalize to 0–1
- `ml/train_mlp.py` — build, train, and save the MLP
- `ml/evaluate.py` — test accuracy, precision, recall, F1, confusion matrix, plots
- `ml/predict_image.py` — predict one image with the saved model
- `ml/download_gtsrb.py` — optional helper to download the official zips
- `models/` — saved model, class names, training history
- `results/` — metrics and graphs
- `data/` — place the dataset here

## 1. Install Python packages

From this project folder:

```bash
python -m pip install -r requirements.txt
```

## 2. Dataset setup (GTSRB, 43 classes)

The dataset is **not bundled** with the project. Official source:

https://benchmark.ini.rub.de/gtsrb_dataset.html

### Option A — automatic download

This downloads the official training images, test images, and test labels into `data/`:

```bash
python -m ml.download_gtsrb
```

After extraction you should have:

```text
data/GTSRB/Final_Training/Images/00000/ ... 00042/
data/GTSRB/Final_Test/Images/
data/GTSRB/Final_Test/GT-final_test.csv   (or data/GT-final_test.csv)
```

### Option B — manual download

Download these official archives and extract them into the `data/` folder:

- `GTSRB_Final_Training_Images.zip`
- `GTSRB_Final_Test_Images.zip`
- `GTSRB_Final_Test_GT.zip` (needed for labeled test accuracy)

The loader also accepts the common Kaggle layout if you already have it:

```text
Train/0 ... Train/42
Train.csv
Test/
Test.csv
```

Put that Kaggle folder under `data/` (or the project root). Do not rename files unless the CSV `Path` values still match.

The loader searches `data/`, `dataset/`, `datasets/`, `GTSRB/`, and the project root. It does not assume a folder exists until it is found.

## 3. Train the MLP

```bash
python -m ml.train_mlp
```

This creates:

- `models/traffic_sign_mlp.keras`
- `models/class_names.json`
- `models/training_history.json`
- `results/metrics.json`
- `results/accuracy_curve.png`
- `results/loss_curve.png`
- `results/confusion_matrix.png`

Optional quicker smoke test (not for final reported accuracy):

```bash
python -m ml.train_mlp --epochs 5 --max-per-class 50
```

## 4. Evaluate again

```bash
python -m ml.evaluate
```

If the official labeled test set is missing, evaluation uses the validation split and records that in `results/metrics.json`. Results are computed from the trained model. They are not estimated by hand.

## 5. Predict one image

```bash
python -m ml.predict_image path\to\sign.ppm
```

If you omit the path, the script uses a real dataset image when the dataset is present:

```bash
python -m ml.predict_image
```

## Model

Feed-forward MLP (not a CNN):

- Input: flattened 32×32×3 RGB image
- Dense 512 + ReLU
- Dropout
- Dense 256 + ReLU
- Dropout
- Dense 43 + softmax

Optimizer: Adam. Loss: sparse categorical cross-entropy. Metric: accuracy. Early stopping on validation loss.
