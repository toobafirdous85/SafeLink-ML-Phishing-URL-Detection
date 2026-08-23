import pandas as pd
import numpy as np

# ============================================================
# 1. Load dataset
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ============================================================
# 2. Remove duplicate URLs
# ============================================================

df = df.drop_duplicates(subset="URL")

print("Shape after removing duplicate URLs:", df.shape)


# ============================================================
# 3. Remove unnecessary/redundant features
# ============================================================

features_to_remove = [
    "FILENAME",
    "NoOfLettersInURL",
    "URLTitleMatchScore"
]

df = df.drop(columns=features_to_remove)

print("\n--- Removed Features ---")
for feature in features_to_remove:
    print(feature)


# ============================================================
# 4. Features selected for log1p transformation
# ============================================================

log_features = [
    "NoOfCSS",
    "NoOfObfuscatedChar",
    "NoOfJS",
    "NoOfEqualsInURL",
    "NoOfEmptyRef",
    "NoOfAmpersandInURL",
    "NoOfiFrame",
    "NoOfDegitsInURL",
    "NoOfPopup",
    "NoOfExternalRef",
    "NoOfSelfRef",
    "LineOfCode",
    "LargestLineLength",
    "NoOfOtherSpecialCharsInURL",
    "NoOfImage",
    "NoOfQMarkInURL",
    "NoOfSubDomain"
]


# ============================================================
# 5. Apply log1p transformation
# ============================================================

print("\n--- Applying log1p transformation ---")

for feature in log_features:
    df[feature] = np.log1p(df[feature])

print("Log1p transformation completed successfully!")


# ============================================================
# 6. Check transformed features
# ============================================================

print("\n--- Transformed Feature Summary ---")

print(df[log_features].describe().T[
    ["mean", "std", "min", "max"]
])


# ============================================================
# 7. Check skewness after transformation
# ============================================================

print("\n--- Skewness After log1p Transformation ---")

transformed_skewness = df[log_features].skew().sort_values(
    ascending=False
)

print(transformed_skewness)


# ============================================================
# 8. Save transformed dataset
# ============================================================

output_file = "data/phiusiil+phishing+url+dataset/transformed_dataset.csv"

df.to_csv(output_file, index=False)

print("\nTransformed dataset saved successfully!")
print("Output file:", output_file)
print("Final shape:", df.shape)
