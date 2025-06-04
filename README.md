
# 🧠 K-Nearest Neighbors (KNN) Classification

This project demonstrates a complete implementation of the **K-Nearest Neighbors (KNN)** algorithm using the **Iris dataset**. The objective is to understand KNN for classification, evaluate its performance across different values of `k`, and visualize decision boundaries.

## 🚀 Features

- Uses **Iris dataset** for multi-class classification
- Normalizes features using `StandardScaler`
- Trains KNN models for various values of `k` (1, 3, 5, 7)
- Evaluates models using:
  - Accuracy score
  - Confusion matrix
- Visualizes decision boundaries for `k=5` using 2D plots

## 📦 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```txt
numpy
pandas
scikit-learn
matplotlib
```

## 🧪 How to Run

```bash
python knn_classifier.py
```

## 📊 Output

- Console output showing **accuracy** and **confusion matrix** for different `k` values.
- A **plot of decision boundaries** when `k=5` using the first two features.

## 📁 Project Structure

```
├── knn_classifier.py     # Main Python script
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

## 📘 Dataset Info

- **Dataset**: Iris (loaded via `sklearn.datasets`)
- **Classes**: Setosa, Versicolor, Virginica
- **Features Used**: Sepal length, Sepal width (for visualization)

## 📷 Demo

![Decision Boundary](https://upload.wikimedia.org/wikipedia/commons/e/e7/Iris_dataset_scatterplot.svg)

> _Note: Image above is a general representation; actual output includes similar decision boundaries generated from your model._
