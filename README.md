# Multiple Linear Regression — Normal Equation (No Frameworks)

A simple and educational implementation of **Multiple Linear Regression** using the **Normal Equation** method in **pure NumPy**. No machine learning libraries or frameworks are used.

## 🧠 What It Does
- Loads a dataset from `ex_2.csv`
- Reorders columns to make target `y` first
- Adds a bias term to the feature matrix
- Solves for weights using the closed-form normal equation:
  
  \[
  \theta = (X^T X)^{-1} X^T y
  \]

- Makes predictions
- Compares actual vs predicted values
- Visualizes results with a scatter plot

## 📁 Files Included
- `code.py`: Full implementation
- `ex_2.csv`: Sample dataset (you can replace with your own)
- `Y_pred VS Y_real using normal equation.png`: Output plot comparing predictions with real values

## ▶️ How to Run
```bash
python code.py
