import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import streamlit as st


housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target
print(df.head())


X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("RMSE:", rmse)
print("R² Score:", r2)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\nSample Predictions:")
print(results.head(11))

st.title("House Price Predictor")
income = st.number_input("Median Income")
rooms = st.number_input("Average Rooms")

if st.button("Predict"):
    prediction = model.predict([[income, rooms, ...]])
    st.write(prediction)