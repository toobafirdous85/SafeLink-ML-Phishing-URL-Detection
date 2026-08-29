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

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 3. Input Data Validation
# ============================================================

print("\n--- Input Data Validation ---")

print("Missing values:", X.isnull().sum().sum())

print(
    "Object columns:",
    X.select_dtypes(include="object").columns.tolist()
)

print(
    "Label present in features:",
    "label" in X.columns
)


# ============================================================
# 4. Train-Test Split
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
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 5. Verify class distribution
# ============================================================

print("\n--- Training Class Distribution (%) ---")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\n--- Testing Class Distribution (%) ---")
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 6. Random Forest Initialization
# ============================================================

print("\n--- Random Forest Initialization ---")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Model: RandomForestClassifier")
print("n_estimators: 100")
print("max_depth: 10")
print("random_state: 42")


# ============================================================
# 7. Model Training
# ============================================================

print("\n--- Model Training ---")

model.fit(X_train, y_train)

print("Random Forest training completed successfully!")


# ============================================================
# 8. Generate Predictions
# ============================================================

print("\n--- Generating Predictions ---")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Predictions generated successfully!")
print("Number of test predictions:", len(y_test_pred))


# ============================================================
# 9. Test Performance
# ============================================================

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\n--- Random Forest Performance ---")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 10. Confusion Matrix
# ============================================================

print("\n--- Confusion Matrix ---")

cm = confusion_matrix(y_test, y_test_pred)

print(cm)


# ============================================================
# 11. Classification Report
# ============================================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_test_pred
    )
)


# ============================================================
# 12. Model Structure
# ============================================================

print("\n--- Random Forest Information ---")

print("Number of trees:", model.n_estimators)

print(
    "Maximum tree depth:",
    model.max_depth
)


# ============================================================
# 13. Feature Importance Analysis
# ============================================================

print("\n--- Feature Importance Analysis ---")

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 15 Most Important Features:")

print(
    feature_importance.head(15)
)


# ============================================================
# 14. Count Important Features
# ============================================================

non_zero_features = (
    feature_importance > 0
).sum()

print(
    "\nNumber of features with non-zero importance:",
    non_zero_features
)


# ============================================================
# 15. Training Performance
# ============================================================

training_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\n--- Training Performance ---")

print(
    f"Training Accuracy: {training_accuracy:.4f}"
)


# ============================================================
# 16. Generalization Check
# ============================================================

generalization_gap = (
    training_accuracy - accuracy
)

print("\n--- Generalization Check ---")

print(
    f"Training Accuracy: {training_accuracy:.4f}"
)

print(
    f"Testing Accuracy : {accuracy:.4f}"
)

print(
    f"Generalization Gap: {generalization_gap:.4f}"
)


# ============================================================
# 17. Final Validation
# ============================================================

print("\n--- Final Validation ---")

print(
    "Training missing values:",
    X_train.isnull().sum().sum()
)

print(
    "Testing missing values:",
    X_test.isnull().sum().sum()
)

print(
    "Training object columns:",
    X_train.select_dtypes(
        include="object"
    ).columns.tolist()
)

print(
    "Testing object columns:",
    X_test.select_dtypes(
        include="object"
    ).columns.tolist()
)

print(
    "Label present in X_train:",
    "label" in X_train.columns
)

print(
    "Label present in X_test:",
    "label" in X_test.columns
)


print("\nRandom Forest baseline experiment completed successfully!")