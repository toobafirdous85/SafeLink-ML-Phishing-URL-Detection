import pandas as pd

# ============================================================
# 1. Load transformed dataset
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/transformed_dataset.csv"

df = pd.read_csv(file_path)

print("Transformed dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. Analyze categorical features
# ============================================================

categorical_features = df.select_dtypes(
    include=["object"]
).columns.tolist()

print("\n--- Object/Text Features ---")
print(categorical_features)


# ============================================================
# 3. Unique values
# ============================================================

print("\n--- Unique Values ---")

for feature in categorical_features:
    print(
        f"{feature}: {df[feature].nunique()} unique values"
    )


# ============================================================
# 4. TLD frequency
# ============================================================

print("\n--- Top 20 TLDs ---")

print(
    df["TLD"]
    .value_counts()
    .head(20)
)


# ============================================================
# 5. Rare TLDs
# ============================================================

tld_counts = df["TLD"].value_counts()

rare_tlds = tld_counts[tld_counts < 10]

print("\n--- Rare TLDs (<10 occurrences) ---")
print("Number of rare TLD categories:", len(rare_tlds))


# ============================================================
# 6. Coverage of top TLDs
# ============================================================

for n in [10, 20, 50, 100]:
    coverage = (
        tld_counts.head(n).sum()
        / len(df)
        * 100
    )

    print(
        f"Top {n} TLDs cover: {coverage:.2f}% of dataset"
    )


print("\nCategorical analysis completed successfully!")
