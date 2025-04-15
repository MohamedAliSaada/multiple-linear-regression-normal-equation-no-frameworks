![Y_pred vs Y_real](Y_pred%20VS%20Y_real%20using%20normal%20equation.png)

# Multiple Linear Regression — Normal Equation (No Frameworks)

A simple implementation of **Multiple Linear Regression** using the **Normal Equation** in pure **NumPy**, without any machine learning frameworks.

## 🔍 What It Does
- Loads a dataset from `ex_2.csv`
- Adds a bias (intercept) term to the features
- Solves for weights using the closed-form normal equation:
  
  \[
  \theta = (X^T X)^{-1} X^T y
  \]

- Predicts target values
- Compares actual vs predicted values
- Visualizes results in a scatter plot

## ▶️ How to Run

Make sure you have Python installed with `numpy`, `pandas`, and `matplotlib`:

```bash
pip install numpy pandas matplotlib
python code.py
