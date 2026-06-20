import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("dataset.csv")

# Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Linear Regression
# ==========================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

accuracy_lr = r2_lr * 100

print("\n===== Linear Regression =====")
print("Accuracy Score:", round(accuracy_lr, 2), "%")
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("R² Score:", r2_lr)

# ==========================
# Decision Tree
# ==========================
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

accuracy_dt = r2_dt * 100

print("\n===== Decision Tree Regressor =====")
print("Accuracy Score:", round(accuracy_dt, 2), "%")
print("MAE:", mae_dt)
print("MSE:", mse_dt)
print("R² Score:", r2_dt)

# ==========================
# Best Model
# ==========================
print("\n===== Best Model =====")

if r2_dt > r2_lr:
    print("Decision Tree Regressor")
else:
    print("Linear Regression")