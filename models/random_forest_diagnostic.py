import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. Load encoded dataset
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. Separate features and target
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print("\n--- Feature / Target Separation ---")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 3. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)


# ============================================================
# 4. Baseline Random Forest
# ============================================================

print("\n--- Baseline Random Forest ---")

baseline_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)


# ============================================================
# 5. Baseline Performance
# ============================================================

baseline_accuracy = accuracy_score(
    y_test,
    baseline_test_pred
)

baseline_precision = precision_score(
    y_test,
    baseline_test_pred
)

baseline_recall = recall_score(
    y_test,
    baseline_test_pred
)

baseline_f1 = f1_score(
    y_test,
    baseline_test_pred
)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred
)

print("\n--- Baseline Performance ---")

print(f"Accuracy : {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall   : {baseline_recall:.4f}")
print(f"F1-score : {baseline_f1:.4f}")

print("\nBaseline Training Accuracy:")
print(f"{baseline_train_accuracy:.4f}")

print("\n--- Baseline Confusion Matrix ---")
print(confusion_matrix(y_test, baseline_test_pred))


# ============================================================
# 6. Investigate URLSimilarityIndex
# ============================================================

print("\n--- Dominant Feature Investigation ---")

print("Feature: URLSimilarityIndex")

print("\nFeature statistics by class:")

print(
    df.groupby("label")["URLSimilarityIndex"]
    .agg(["count", "mean", "std", "min", "max"])
)


# ============================================================
# 7. Remove URLSimilarityIndex
# ============================================================

print("\n--- Experiment: Removing URLSimilarityIndex ---")

X_reduced = X.drop(
    columns=["URLSimilarityIndex"]
)

print(
    "Reduced feature matrix shape:",
    X_reduced.shape
)


# ============================================================
# 8. Train-Test Split Without Dominant Feature
# ============================================================

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nReduced Training features:", X_train_reduced.shape)
print("Reduced Testing features:", X_test_reduced.shape)


# ============================================================
# 9. Random Forest Without URLSimilarityIndex
# ============================================================

reduced_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("\n--- Training Reduced Random Forest ---")

reduced_model.fit(
    X_train_reduced,
    y_train_reduced
)

print(
    "Reduced Random Forest training completed successfully!"
)


# ============================================================
# 10. Generate Reduced Model Predictions
# ============================================================

reduced_train_pred = reduced_model.predict(
    X_train_reduced
)

reduced_test_pred = reduced_model.predict(
    X_test_reduced
)


# ============================================================
# 11. Reduced Model Performance
# ============================================================

reduced_accuracy = accuracy_score(
    y_test_reduced,
    reduced_test_pred
)

reduced_precision = precision_score(
    y_test_reduced,
    reduced_test_pred
)

reduced_recall = recall_score(
    y_test_reduced,
    reduced_test_pred
)

reduced_f1 = f1_score(
    y_test_reduced,
    reduced_test_pred
)

reduced_train_accuracy = accuracy_score(
    y_train_reduced,
    reduced_train_pred
)


print("\n--- Performance Without URLSimilarityIndex ---")

print(f"Accuracy : {reduced_accuracy:.4f}")
print(f"Precision: {reduced_precision:.4f}")
print(f"Recall   : {reduced_recall:.4f}")
print(f"F1-score : {reduced_f1:.4f}")


# ============================================================
# 12. Reduced Model Confusion Matrix
# ============================================================

print("\n--- Confusion Matrix Without URLSimilarityIndex ---")

print(
    confusion_matrix(
        y_test_reduced,
        reduced_test_pred
    )
)


# ============================================================
# 13. Reduced Model Classification Report
# ============================================================

print("\n--- Classification Report Without URLSimilarityIndex ---")

print(
    classification_report(
        y_test_reduced,
        reduced_test_pred
    )
)


# ============================================================
# 14. Generalization Check
# ============================================================

reduced_gap = (
    reduced_train_accuracy -
    reduced_accuracy
)

print("\n--- Generalization Check ---")

print(
    f"Training Accuracy: {reduced_train_accuracy:.4f}"
)

print(
    f"Testing Accuracy : {reduced_accuracy:.4f}"
)

print(
    f"Generalization Gap: {reduced_gap:.4f}"
)


# ============================================================
# 15. Feature Importance Without URLSimilarityIndex
# ============================================================

print("\n--- Feature Importance Without URLSimilarityIndex ---")

reduced_importance = pd.Series(
    reduced_model.feature_importances_,
    index=X_reduced.columns
).sort_values(ascending=False)

print("\nTop 15 Most Important Features:")

print(
    reduced_importance.head(15)
)


# ============================================================
# 16. Count Non-Zero Important Features
# ============================================================

non_zero_features = (
    reduced_importance > 0
).sum()

print(
    "\nNumber of features with non-zero importance:",
    non_zero_features
)


# ============================================================
# 17. Performance Comparison
# ============================================================

print("\n--- Baseline vs Reduced Model ---")

comparison = pd.DataFrame({
    "Baseline": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],
    "Without_URLSimilarityIndex": [
        reduced_accuracy,
        reduced_precision,
        reduced_recall,
        reduced_f1
    ]
}, index=[
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
])

print(comparison.round(4))


# ============================================================
# 18. Performance Difference
# ============================================================

print("\n--- Performance Difference ---")

print(
    f"Accuracy change : "
    f"{reduced_accuracy - baseline_accuracy:.4f}"
)

print(
    f"Precision change: "
    f"{reduced_precision - baseline_precision:.4f}"
)

print(
    f"Recall change   : "
    f"{reduced_recall - baseline_recall:.4f}"
)

print(
    f"F1-score change : "
    f"{reduced_f1 - baseline_f1:.4f}"
)


# ============================================================
# 19. Final Conclusion
# ============================================================

print("\n--- Diagnostic Experiment Completed ---")

print(
    "The baseline Random Forest was compared "
    "with a model trained without URLSimilarityIndex."
)

print(
    "This experiment measures the model's dependence "
    "on the dominant feature and evaluates whether "
    "other engineered features retain predictive power."
)