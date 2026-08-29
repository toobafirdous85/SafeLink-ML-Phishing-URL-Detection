import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path =  "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 2. FEATURE / TARGET SEPARATION
# ============================================================

X = df.drop(columns=["label"])
y = df["label"]

print("\n--- Feature / Target Separation ---")
print("Features:", X.shape)
print("Target:", y.shape)

# ============================================================
# 3. TRAIN / TEST SPLIT
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

# ============================================================
# FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(name, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed!")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Performance ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Confusion matrix
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    # Training performance
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)

    print("\n--- Generalization Check ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy : {accuracy:.4f}")
    print(f"Generalization Gap: {train_accuracy - accuracy:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Training Accuracy": train_accuracy
    }


# ============================================================
# MODEL A — FULL FEATURE SET
# ============================================================

results = []

results.append(
    evaluate_model(
        "MODEL A — FULL FEATURE SET",
        X_train,
        X_test,
        y_train,
        y_test
    )
)

# ============================================================
# MODEL B — REMOVE URLSimilarityIndex
# ============================================================

print("\n\n--- MODEL B: Removing URLSimilarityIndex ---")

X_no_similarity = X.drop(
    columns=["URLSimilarityIndex"]
)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_no_similarity,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Features after removal:", X_no_similarity.shape)

results.append(
    evaluate_model(
        "MODEL B — WITHOUT URLSimilarityIndex",
        X_train_b,
        X_test_b,
        y_train_b,
        y_test_b
    )
)

# ============================================================
# MODEL C — REMOVE SUSPICIOUS FEATURES
# ============================================================

suspicious_features = [
    "URLSimilarityIndex",
    "NoOfSelfRef",
    "NoOfImage",
    "LineOfCode",
    "NoOfExternalRef",
    "NoOfJS",
    "HasSocialNet",
    "NoOfCSS",
    "HasCopyrightInfo",
    "HasDescription",
    "IsHTTPS",
    "NoOfOtherSpecialCharsInURL",
    "DomainTitleMatchScore",
    "HasSubmitButton",
    "IsResponsive",
    "SpacialCharRatioInURL",
    "LargestLineLength",
    "HasHiddenFields",
    "NoOfDegitsInURL"
]

# Only remove columns that actually exist
features_to_remove = [
    feature
    for feature in suspicious_features
    if feature in X.columns
]

print("\n\n--- MODEL C: Removing Suspicious Features ---")

print("Features selected for removal:")

for feature in features_to_remove:
    print("-", feature)

X_reduced = X.drop(
    columns=features_to_remove
)

print("\nOriginal feature count:", X.shape[1])
print("Reduced feature count :", X_reduced.shape[1])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_reduced,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

results.append(
    evaluate_model(
        "MODEL C — WITHOUT SUSPICIOUS FEATURES",
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c
    )
)

# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL ABLATION STUDY COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Training Accuracy"
        ]
    ].to_string(index=False)
)

# ============================================================
# PERFORMANCE CHANGE
# ============================================================

baseline_f1 = results[0]["F1-score"]

print("\n--- F1 Performance Change From Baseline ---")

for result in results:

    difference = result["F1-score"] - baseline_f1

    print(
        f"{result['Model']}: "
        f"{difference:+.4f}"
    )

print("\n" + "=" * 70)
print("ABLATION STUDY COMPLETED")
print("=" * 70)

print(
    "\nThis experiment evaluates whether Random Forest "
    "performance remains strong after removing highly "
    "predictive and potentially dataset-specific features."
)