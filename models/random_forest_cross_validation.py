import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD ENCODED DATASET
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# FEATURE / TARGET SEPARATION
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print("\n--- Dataset Preparation ---")
print("Features:", X.shape)
print("Target:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# INPUT VALIDATION
# ============================================================

print("\n--- Input Validation ---")

print("Missing values:", X.isnull().sum().sum())
print("Object columns:", X.select_dtypes(include=["object"]).columns.tolist())
print("Label present in features:", "label" in X.columns)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n--- 5-Fold Cross Validation ---")
print("Number of folds:", cv.n_splits)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)


# ============================================================
# FOLD-LEVEL RESULTS
# ============================================================

print("\n--- Fold-Level Results ---")

for i in range(5):

    print(f"\nFold {i + 1}:")
    print(f"Accuracy : {results['test_accuracy'][i]:.4f}")
    print(f"Precision: {results['test_precision'][i]:.4f}")
    print(f"Recall   : {results['test_recall'][i]:.4f}")
    print(f"F1-score : {results['test_f1'][i]:.4f}")


# ============================================================
# CROSS-VALIDATION SUMMARY
# ============================================================

print("\n--- Cross-Validation Summary ---")

print(
    f"Accuracy  : "
    f"{results['test_accuracy'].mean():.4f} ± "
    f"{results['test_accuracy'].std():.4f}"
)

print(
    f"Precision : "
    f"{results['test_precision'].mean():.4f} ± "
    f"{results['test_precision'].std():.4f}"
)

print(
    f"Recall    : "
    f"{results['test_recall'].mean():.4f} ± "
    f"{results['test_recall'].std():.4f}"
)

print(
    f"F1        : "
    f"{results['test_f1'].mean():.4f} ± "
    f"{results['test_f1'].std():.4f}"
)


# ============================================================
# TRAINING VS VALIDATION PERFORMANCE
# ============================================================

print("\n--- Training vs Validation Performance ---")

train_accuracy = results["train_accuracy"].mean()
test_accuracy = results["test_accuracy"].mean()

print(f"Mean Training Accuracy   : {train_accuracy:.4f}")
print(f"Mean Validation Accuracy : {test_accuracy:.4f}")

print(
    f"Cross-Validation Gap     : "
    f"{train_accuracy - test_accuracy:.4f}"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n============================================================")
print("RANDOM FOREST CROSS-VALIDATION COMPLETED")
print("============================================================")

print(
    "\nThe 5-fold stratified cross-validation provides a more robust "
    "estimate of Random Forest performance across multiple data splits."
)