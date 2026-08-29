import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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

print("\n--- Feature / Target Separation ---")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# INPUT VALIDATION
# ============================================================

print("\n--- Input Validation ---")

print("Missing values:", X.isnull().sum().sum())
print("Object columns:", X.select_dtypes(include=["object"]).columns.tolist())
print("Label present in features:", "label" in X.columns)


# ============================================================
# TRAIN / TEST SPLIT
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
# MODEL DEFINITIONS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}


# ============================================================
# MODEL COMPARISON
# ============================================================

results = []


for model_name, model in models.items():

    print("\n")
    print("=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    print("\n--- Training ---")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print("Training completed successfully!")

    print("\n--- Generating Predictions ---")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully!")
    print("Number of predictions:", len(y_pred))

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Performance ---")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Confusion Matrix ---")
    print(cm)

    # --------------------------------------------------------
    # TRAINING PERFORMANCE
    # --------------------------------------------------------

    y_train_pred = model.predict(X_train)

    training_accuracy = accuracy_score(
        y_train,
        y_train_pred
    )

    generalization_gap = (
        training_accuracy - accuracy
    )

    print("\n--- Generalization Check ---")

    print(
        f"Training Accuracy: "
        f"{training_accuracy:.4f}"
    )

    print(
        f"Testing Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Generalization Gap: "
        f"{generalization_gap:.4f}"
    )

    # --------------------------------------------------------
    # TRAINING TIME
    # --------------------------------------------------------

    print(
        f"\nTraining Time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Training Accuracy": training_accuracy,
        "Generalization Gap": generalization_gap,
        "Training Time (seconds)": training_time
    })


# ============================================================
# FINAL MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 85)
print("FINAL MODEL COMPARISON")
print("=" * 85)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# BEST MODEL IDENTIFICATION
# ============================================================

best_model = results_df.loc[
    results_df["F1-score"].idxmax()
]

print("\n")
print("=" * 65)
print("BEST MODEL")
print("=" * 65)

print(
    "Model:",
    best_model["Model"]
)

print(
    f"Accuracy : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall   : {best_model['Recall']:.4f}"
)

print(
    f"F1-score : {best_model['F1-score']:.4f}"
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

output_file = "data/phiusiil+phishing+url+dataset/model_comparison_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n--- Results Saved ---")
print(
    f"Model comparison results saved to: "
    f"{output_file}"
)


# ============================================================
# COMPLETION
# ============================================================

print("\n")
print("=" * 65)
print("MODEL COMPARISON COMPLETED")
print("=" * 65)

