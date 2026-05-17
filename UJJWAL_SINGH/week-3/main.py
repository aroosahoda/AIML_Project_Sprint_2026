from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ( ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def evaluate_model(model_name, y_true, y_pred):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


base_dir = Path(__file__).resolve().parent
dataset_path = base_dir / "telecom_churn.csv"
output_dir = base_dir / "artifacts"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(dataset_path)

if "Churn" not in df.columns:
    raise ValueError("Expected target column 'Churn' was not found in telecom_churn.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

print("Dataset shape:", df.shape)
class_distribution = y.value_counts().sort_index()
class_percentage = (class_distribution / len(y) * 100).round(2)
print("Class distribution:")
for cls in class_distribution.index:
    print(f"  Class {cls}: {class_distribution[cls]} ({class_percentage[cls]}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Use class weighting to reduce bias toward the majority (non-churn) class.
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
)
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    n_jobs=-1,
)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred)

metrics_df = pd.DataFrame([dt_metrics, rf_metrics]).set_index("model")
metrics_path = output_dir / "model_metrics.csv"
metrics_df.to_csv(metrics_path)

print("\nModel comparison:")
print(metrics_df.round(4))

print("\nClassification report (Decision Tree):")
print(classification_report(y_test, dt_pred, digits=4, zero_division=0))
print("Classification report (Random Forest):")
print(classification_report(y_test, rf_pred, digits=4, zero_division=0))



# Save confusion matrices side by side for both decison tree and random forest
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred)).plot(ax=axes[0], colorbar=False)

axes[0].set_title("Decision Tree Confusion Matrix")
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred)).plot(ax=axes[1], colorbar=False)

axes[1].set_title("Random Forest Confusion Matrix")
plt.tight_layout()

cm_path = output_dir / "confusion_matrices.png"
plt.savefig(cm_path, dpi=200)
plt.close(fig)



# Visualize the tree 
fig = plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.title("Decision Tree Visualization")
tree_path = output_dir / "decision_tree.png"
plt.tight_layout()
plt.savefig(tree_path, dpi=200)
plt.close(fig)

importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importance_path = output_dir / "feature_importance.csv"
importances.to_csv(importance_path, header=["importance"])

fig = plt.figure(figsize=(10, 6))
importances.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
fi_plot_path = output_dir / "feature_importance.png"
plt.savefig(fi_plot_path, dpi=200)
plt.close(fig)

print("\nSaved artifacts:")
print(f"  - {metrics_path}")
print(f"  - {cm_path}")
print(f"  - {tree_path}")
print(f"  - {importance_path}")
print(f"  - {fi_plot_path}")

