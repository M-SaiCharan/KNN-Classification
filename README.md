
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

## 🧪 How to Run

```bash
python main.py
```

## 📊 Output

- Console output showing **accuracy** and **confusion matrix** for different `k` values.
- A **plot of decision boundaries** when `k=5` using the first two features.

## 📘 Dataset Info

- **Dataset**: Iris (loaded via `sklearn.datasets`)
- **Classes**: Setosa, Versicolor, Virginica
- **Features Used**: Sepal length, Sepal width (for visualization)

## 📷 Sample Output

![Figure_1](https://github.com/user-attachments/assets/e3559f2d-d2ca-4b8c-82a3-ab482a5a342c)


