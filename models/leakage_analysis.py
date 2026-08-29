import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# 1. LOAD ENCODED DATASET
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

target_column = "label"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\n--- Feature / Target Separation ---")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

# ============================================================
# 3. BASIC DATA VALIDATION
# ============================================================

print("\n--- Basic Validation ---")

print("Missing values:", X.isnull().sum().sum())
print("Object columns:", X.select_dtypes(include="object").columns.tolist())
print("Label present in features:", target_column in X.columns)

# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n--- Target Distribution ---")

print(y.value_counts())
print("\nTarget distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2))

# ============================================================
# 5. FEATURE UNIQUENESS ANALYSIS
# ============================================================

print("\n--- Feature Uniqueness Analysis ---")

unique_counts = X.nunique().sort_values()

print("\nFeatures with the fewest unique values:")
print(unique_counts.head(20))

# ============================================================
# 6. PERFECT / NEAR-PERFECT CLASS SEPARATION
# ============================================================

print("\n--- Class Separation Analysis ---")

separation_results = []

for feature in X.columns:

    class_0 = X.loc[y == 0, feature]
    class_1 = X.loc[y == 1, feature]

    mean_0 = class_0.mean()
    mean_1 = class_1.mean()

    std_0 = class_0.std()
    std_1 = class_1.std()

    min_0 = class_0.min()
    max_0 = class_0.max()

    min_1 = class_1.min()
    max_1 = class_1.max()

    separation_results.append({
        "Feature": feature,
        "Class_0_Mean": mean_0,
        "Class_1_Mean": mean_1,
        "Class_0_Std": std_0,
        "Class_1_Std": std_1,
        "Class_0_Min": min_0,
        "Class_0_Max": max_0,
        "Class_1_Min": min_1,
        "Class_1_Max": max_1
    })

separation_df = pd.DataFrame(separation_results)

# Difference between class means
separation_df["Mean_Difference"] = (
    separation_df["Class_1_Mean"] -
    separation_df["Class_0_Mean"]
).abs()

separation_df = separation_df.sort_values(
    by="Mean_Difference",
    ascending=False
)

print("\nTop 15 features by absolute class-mean difference:")
print(separation_df.head(15).to_string(index=False))

# ============================================================
# 7. URLSimilarityIndex INVESTIGATION
# ============================================================

if "URLSimilarityIndex" in X.columns:

    print("\n--- URLSimilarityIndex Investigation ---")

    similarity_stats = df.groupby(target_column)["URLSimilarityIndex"].agg(
        ["count", "mean", "std", "min", "max"]
    )

    print(similarity_stats)

    print("\nUnique URLSimilarityIndex values by class:")

    for label in sorted(y.unique()):
        values = df.loc[
            df[target_column] == label,
            "URLSimilarityIndex"
        ].unique()

        print(
            f"Class {label}: "
            f"{len(values)} unique values"
        )

# ============================================================
# 8. HIGH CORRELATION WITH TARGET
# ============================================================

print("\n--- Target Correlation Analysis ---")

numeric_df = df.select_dtypes(include=np.number)

correlations = numeric_df.corr()["label"].drop("label")

correlations = correlations.abs().sort_values(ascending=False)

print("\nTop 20 features by absolute correlation with label:")
print(correlations.head(20))

# ============================================================
# 9. DUPLICATE RECORD ANALYSIS
# ============================================================

print("\n--- Duplicate Record Analysis ---")

duplicate_rows = df.duplicated().sum()

print("Duplicate complete rows:", duplicate_rows)

# ============================================================
# 10. DUPLICATE URL ANALYSIS
# ============================================================

print("\n--- Duplicate URL Analysis ---")

url_column = None

possible_url_columns = [
    "URL",
    "url",
    "Url"
]

for column in possible_url_columns:
    if column in df.columns:
        url_column = column
        break

if url_column is not None:

    duplicate_urls = df[url_column].duplicated().sum()

    print("URL column:", url_column)
    print("Duplicate URLs:", duplicate_urls)

    # Check whether duplicated URLs have conflicting labels
    url_label_counts = df.groupby(url_column)[target_column].nunique()

    conflicting_urls = (url_label_counts > 1).sum()

    print(
        "URLs appearing with different labels:",
        conflicting_urls
    )

else:

    print("Raw URL column not present in encoded dataset.")
    print("Duplicate URL analysis skipped.")

# ============================================================
# 11. FEATURES WITH PERFECT CLASS CONSTANTS
# ============================================================

print("\n--- Perfect Class-Constant Feature Check ---")

constant_class_features = []

for feature in X.columns:

    class_0_unique = X.loc[y == 0, feature].nunique()
    class_1_unique = X.loc[y == 1, feature].nunique()

    if class_0_unique == 1 or class_1_unique == 1:

        constant_class_features.append({
            "Feature": feature,
            "Class_0_Unique": class_0_unique,
            "Class_1_Unique": class_1_unique
        })

constant_df = pd.DataFrame(constant_class_features)

if not constant_df.empty:
    print(constant_df.to_string(index=False))
else:
    print("No features were constant within a class.")

# ============================================================
# 12. TRAIN/TEST DUPLICATE FEATURE PATTERN CHECK
# ============================================================

print("\n--- Train/Test Split Analysis ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training class distribution:",
    y_train.value_counts(normalize=True).round(4).to_dict()
)

print(
    "Testing class distribution:",
    y_test.value_counts(normalize=True).round(4).to_dict()
)

# ============================================================
# 13. IDENTIFY EXTREMELY SUSPICIOUS FEATURES
# ============================================================

print("\n--- Suspicious Feature Summary ---")

suspicious_features = correlations[
    correlations >= 0.90
]

if len(suspicious_features) > 0:

    print(
        "Features with absolute correlation >= 0.90 with label:"
    )

    print(suspicious_features)

else:

    print(
        "No features have absolute correlation >= 0.90 with label."
    )

# ============================================================
# 14. FINAL DIAGNOSTIC SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LEAKAGE / DATA VALIDITY ANALYSIS COMPLETED")
print("=" * 60)

print("\nKey checks performed:")
print("1. Feature uniqueness")
print("2. Class separation")
print("3. URLSimilarityIndex investigation")
print("4. Feature-target correlation")
print("5. Duplicate row analysis")
print("6. Duplicate URL analysis")
print("7. Class-constant feature detection")
print("8. Train/test distribution validation")

print("\nNo data was modified during this analysis.")