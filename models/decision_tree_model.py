import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
# 3. Train/Test Split
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
# 4. Train Decision Tree
# ============================================================

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ============================================================
# 5. Baseline Performance
# ============================================================

print("\n--- Baseline Decision Tree Performance ---")

print("Accuracy :", f"{accuracy_score(y_test, y_pred):.4f}")
print("Precision:", f"{precision_score(y_test, y_pred):.4f}")
print("Recall   :", f"{recall_score(y_test, y_pred):.4f}")
print("F1-score :", f"{f1_score(y_test, y_pred):.4f}")


# ============================================================
# 6. Tree Structure
# ============================================================

print("\n--- Tree Structure ---")

print("Number of nodes:", model.tree_.node_count)
print("Tree depth:", model.get_depth())

print("\n--- Learned Decision Rules ---")

rules = export_text(
    model,
    feature_names=list(X.columns),
    decimals=4
)

print(rules)


# ============================================================
# 7. Feature Importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n--- Feature Importance ---")
print(importance.head(15))


# ============================================================
# 8. Dominant Feature Investigation
# ============================================================

print("\n--- Dominant Feature Investigation ---")

dominant_feature = "URLSimilarityIndex"

print(
    "Dominant feature:",
    dominant_feature
)

print(
    "\nFeature statistics by class:"
)

print(
    df.groupby("label")[dominant_feature].agg(
        ["count", "mean", "std", "min", "max"]
    )
)


# ============================================================
# 9. Remove Dominant Feature
# ============================================================

print("\n--- Experiment: Removing URLSimilarityIndex ---")

X_without_similarity = X.drop(
    columns=[dominant_feature]
)

X_train_no_sim, X_test_no_sim, y_train_no_sim, y_test_no_sim = train_test_split(
    X_without_similarity,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model_no_sim = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

model_no_sim.fit(
    X_train_no_sim,
    y_train_no_sim
)

y_pred_no_sim = model_no_sim.predict(
    X_test_no_sim
)


# ============================================================
# 10. Performance Without Dominant Feature
# ============================================================

print("\n--- Performance Without URLSimilarityIndex ---")

print(
    "Accuracy :",
    f"{accuracy_score(y_test_no_sim, y_pred_no_sim):.4f}"
)

print(
    "Precision:",
    f"{precision_score(y_test_no_sim, y_pred_no_sim):.4f}"
)

print(
    "Recall   :",
    f"{recall_score(y_test_no_sim, y_pred_no_sim):.4f}"
)

print(
    "F1-score :",
    f"{f1_score(y_test_no_sim, y_pred_no_sim):.4f}"
)

print(
    "\nTree depth without URLSimilarityIndex:",
    model_no_sim.get_depth()
)

print(
    "Number of nodes without URLSimilarityIndex:",
    model_no_sim.tree_.node_count
)


# ============================================================
# 11. Diagnostic Conclusion
# ============================================================

print("\n--- Diagnostic Experiment Completed ---")
print(
    "Comparison between the baseline tree and the "
    "feature-removed tree can be used to assess "
    "dependence on URLSimilarityIndex."
)