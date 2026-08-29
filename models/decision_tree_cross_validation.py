import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

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

print("\n--- Dataset Preparation ---")
print("Features:", X.shape)
print("Target:", y.shape)


# ============================================================
# 3. Define Decision Tree
# ============================================================

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)


# ============================================================
# 4. Stratified 5-Fold Cross Validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n--- 5-Fold Cross Validation ---")
print("Number of folds:", cv.n_splits)


scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)


# ============================================================
# 5. Display Fold Results
# ============================================================

print("\n--- Fold-Level Results ---")

for i in range(5):

    print(
        f"\nFold {i + 1}:"
    )

    print(
        "Accuracy :",
        f"{results['test_accuracy'][i]:.4f}"
    )

    print(
        "Precision:",
        f"{results['test_precision'][i]:.4f}"
    )

    print(
        "Recall   :",
        f"{results['test_recall'][i]:.4f}"
    )

    print(
        "F1-score :",
        f"{results['test_f1'][i]:.4f}"
    )


# ============================================================
# 6. Mean and Standard Deviation
# ============================================================

print("\n--- Cross-Validation Summary ---")

for metric in scoring:

    values = results[f"test_{metric}"]

    print(
        f"{metric.capitalize():10s}: "
        f"{values.mean():.4f} ± {values.std():.4f}"
    )


# ============================================================
# 7. Final Conclusion
# ============================================================

print("\n--- Cross-Validation Completed ---")
print(
    "Cross-validation provides a more robust estimate "
    "of Decision Tree performance across multiple "
    "stratified data splits."
)