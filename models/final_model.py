import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD ENCODED DATASET
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURE / TARGET SEPARATION
# ============================================================

X = df.drop("label", axis=1)
y = df["label"]

print("\n--- Feature / Target Separation ---")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

print("\n--- Input Validation ---")

print("Missing values:", X.isnull().sum().sum())
print("Object columns:", X.select_dtypes(include="object").columns.tolist())
print("Label present in features:", "label" in X.columns)


# ============================================================
# 4. TRAIN / TEST SPLIT
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
print("Testing features :", X_test.shape)
print("Training target  :", y_train.shape)
print("Testing target   :", y_test.shape)


# ============================================================
# 5. INITIALIZE FINAL MODEL
# ============================================================

print("\n--- Final Model Initialization ---")

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

print("Model: DecisionTreeClassifier")
print("max_depth:", 10)
print("random_state:", 42)


# ============================================================
# 6. TRAIN FINAL MODEL
# ============================================================

print("\n--- Final Model Training ---")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Final Decision Tree training completed successfully!")
print(f"Training Time: {training_time:.4f} seconds")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\n--- Generating Predictions ---")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")
print("Number of predictions:", len(y_pred))


# ============================================================
# 8. FINAL PERFORMANCE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Final Model Performance ---")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n--- Final Confusion Matrix ---")
print(cm)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n--- Final Classification Report ---")
print(classification_report(y_test, y_pred))


# ============================================================
# 11. TRAINING PERFORMANCE
# ============================================================

y_train_pred = model.predict(X_train)

training_accuracy = accuracy_score(y_train, y_train_pred)

generalization_gap = training_accuracy - accuracy

print("\n--- Generalization Check ---")

print(f"Training Accuracy: {training_accuracy:.4f}")
print(f"Testing Accuracy : {accuracy:.4f}")
print(f"Generalization Gap: {generalization_gap:.4f}")


# ============================================================
# 12. FINAL TREE INFORMATION
# ============================================================

print("\n--- Final Model Structure ---")

print("Number of tree nodes:", model.tree_.node_count)
print("Tree depth:", model.tree_.max_depth)
print("Number of input features:", X.shape[1])


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n--- Top 15 Final Feature Importances ---")
print(feature_importance.head(15))


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

output_directory = "data/phiusiil+phishing+url+dataset"

os.makedirs(output_directory, exist_ok=True)

model_file = os.path.join(
    output_directory,
    "final_decision_tree_model.joblib"
)

joblib.dump(model, model_file)

print("\n--- Model Saved ---")
print("Final model saved to:", model_file)


# ============================================================
# 15. SAVE FEATURE NAMES
# ============================================================

feature_file = os.path.join(
    output_directory,
    "final_model_features.csv"
)

pd.DataFrame({
    "feature": X.columns
}).to_csv(feature_file, index=False)

print("Feature list saved to:", feature_file)


# ============================================================
# 16. SAVE FINAL METRICS
# ============================================================

metrics_file = os.path.join(
    output_directory,
    "final_model_metrics.csv"
)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Training Accuracy",
        "Generalization Gap",
        "Training Time (seconds)",
        "Tree Depth",
        "Tree Nodes",
        "Number of Features"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        training_accuracy,
        generalization_gap,
        training_time,
        model.tree_.max_depth,
        model.tree_.node_count,
        X.shape[1]
    ]
})

metrics.to_csv(metrics_file, index=False)

print("Final metrics saved to:", metrics_file)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

predictions_file = os.path.join(
    output_directory,
    "final_model_predictions.csv"
)

predictions = pd.DataFrame({
    "Actual_Label": y_test.values,
    "Predicted_Label": y_pred,
    "Phishing_Probability": y_prob
})

predictions.to_csv(predictions_file, index=False)

print("Test predictions saved to:", predictions_file)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nSelected Model: Decision Tree")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nModel artifacts:")
print("1. final_decision_tree_model.joblib")
print("2. final_model_features.csv")
print("3. final_model_metrics.csv")
print("4. final_model_predictions.csv")

print("\nThe final Decision Tree model is ready for deployment.")
print("=" * 70)