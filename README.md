# Image Classification with PCA and KNN

An educational image-classification pipeline that combines Principal Component Analysis (PCA) for dimensionality reduction with K-Nearest Neighbors (KNN) for prediction.

The project demonstrates a complete classical machine-learning workflow: loading a directory-based image dataset, preprocessing images, splitting the data, reducing dimensionality, training a classifier, evaluating predictions, saving the fitted pipeline, and using it to classify a new image.

## Pipeline

1. Load images from one subdirectory per class.
2. Convert each image to grayscale and resize it to `64 x 64` pixels.
3. Flatten the pixels and standardize the features with `StandardScaler`.
4. Create a stratified 80/20 train-test split.
5. Reduce the feature space with up to 50 PCA components.
6. Train a KNN classifier with `k = 5`.
7. Report accuracy and per-class precision, recall, and F1-score.
8. Save the preprocessing objects, model, labels, and target image size with Joblib.

## Repository structure

```text
.
|-- modeltrain.ipynb     # Training, evaluation, and model export
|-- clasifica.ipynb      # Inference on a single image
|-- saved_models/        # Serialized scaler, PCA, KNN, labels, and image size
`-- README.md
```

## Dataset layout

The dataset is not included. Create a `dataset_imagenes` directory with one folder per class:

```text
dataset_imagenes/
|-- class_a/
|   |-- image_001.jpg
|   `-- image_002.jpg
`-- class_b/
    |-- image_001.jpg
    `-- image_002.jpg
```

Supported formats are PNG, JPG, JPEG, GIF, and BMP. Use enough examples in every class to support a stratified train-test split.

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Run `modeltrain.ipynb` first. To classify a new image, update `new_image_path` in `clasifica.ipynb` and then run that notebook.

## Reproducibility notes

- Reported performance depends entirely on the dataset and class balance; this repository does not claim a benchmark result.
- The saved Joblib files are version-dependent. Retrain the pipeline if they are incompatible with your local scikit-learn version.
- Joblib and pickle-based files can execute code while loading. Only load model artifacts from sources you trust.
- For a stronger experiment, add cross-validation, tune `k` and the number of PCA components, and include a confusion matrix.

## Technologies

Python, NumPy, OpenCV, scikit-learn, Matplotlib, Joblib, and Jupyter Notebook.
