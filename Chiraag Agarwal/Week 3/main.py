import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv("train.csv")
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]

X = data[features]
y = data["Survived"]
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Sex"] = X["Sex"].map({
    "male": 0,
    "female": 1
})
X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print("\nFeature Importance:")
print(importance)
