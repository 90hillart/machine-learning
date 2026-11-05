# House Price Prediction using Linear Regression

This repository contains a simple machine learning project that implements a **Linear Regression** model using `scikit-learn` to predict house prices based on house sizes.

## 🚀 Overview

The model uses a sample dataset of house sizes (in square feet) and their corresponding prices to train a linear model and predict prices for unseen house sizes.

### Features
* **Data Visualization**: Interactive scatter plots showing the relationship between house size and price.
* **Regression Modeling**: Trained using the Ordinary Least Squares method via `scikit-learn`'s `LinearRegression`.
* **Model Evaluation**: Metrics computed including Mean Squared Error (MSE) and R² Score to measure performance.
* **Visualization of the Fit**: Plotting the regression line over the dataset points.

---

## 🛠️ Prerequisites & Installation

Ensure you have Python installed. You can install the required packages using `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 💻 Usage

To run the script and train/visualize the model, execute:

```bash
python main.py
```

### Script Output
1. A scatter plot of the original data points (Size vs. Price).
2. The predicted prices for the test set.
3. Performance metrics printed in the console:
   - **MSE** (Mean Squared Error)
   - **R² Score**
   - **Slope & Intercept** of the regression line
4. A final plot illustrating the linear regression model fit.
