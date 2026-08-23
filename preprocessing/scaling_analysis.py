import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Load transformed dataset
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/transformed_dataset.csv"

df = pd.read_csv(file_path)

print("Transformed dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. Identify numerical features
# ============================================================

numeric_features = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Remove target variable
numeric_features.remove("label")

print("\n--- Number of numerical features ---")
print(len(numeric_features))

print("\n--- Numerical features ---")
print(numeric_features)


# ============================================================
# 3. Check feature ranges before scaling
# ============================================================

print("\n--- Feature Range Before Scaling ---")

range_before = pd.DataFrame({
    "Minimum": df[numeric_features].min(),
    "Maximum": df[numeric_features].max()
})

print(range_before)


# ============================================================
# 4. Apply StandardScaler
# ============================================================

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    df[numeric_features]
)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=numeric_features
)


# ============================================================
# 5. Check scaled feature statistics
# ============================================================

print("\n--- Mean After Standardization ---")

print(
    scaled_df.mean().round(4)
)


print("\n--- Standard Deviation After Standardization ---")

print(
    scaled_df.std().round(4)
)


# ============================================================
# 6. Verify scaling
# ============================================================

print("\n--- Scaling Verification ---")

print(
    "Maximum absolute mean:",
    abs(scaled_df.mean()).max()
)

print(
    "Average standard deviation:",
    scaled_df.std().mean()
)


print("\nStandardization completed successfully!")
