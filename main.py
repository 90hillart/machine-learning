import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Size': [500, 700, 900, 1100, 1300, 1500, 1700, 1900],
    'Price': [100000, 150000, 180000, 250000, 300000, 340000, 390000, 450000]
}

df = pd.DataFrame(data)

print(df)
plt.scatter(df['Size'], df['Price'])
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('House Size vs Price')
plt.show()
X = df[['Size']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')

plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('Linear Regression Model')
plt.show()