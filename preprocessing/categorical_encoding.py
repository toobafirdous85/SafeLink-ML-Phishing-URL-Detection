import pandas as pd

# ============================================================
# 1. Load transformed dataset
# ============================================================

file_path = "data/phiusiil+phishing+url+dataset/transformed_dataset.csv"

df = pd.read_csv(file_path)

print("Transformed dataset loaded successfully!")
print("Original shape:", df.shape)


# ============================================================
# 2. Identify common TLDs
# ============================================================

tld_counts = df["TLD"].value_counts()

common_tlds = tld_counts.head(50).index

print("\nNumber of common TLDs retained:", len(common_tlds))


# ============================================================
# 3. Group rare TLDs into OTHER
# ============================================================

df["TLD_grouped"] = df["TLD"].where(
    df["TLD"].isin(common_tlds),
    "OTHER"
)


# ============================================================
# 4. Check resulting categories
# ============================================================

print("\n--- TLD Categories After Grouping ---")

print(
    df["TLD_grouped"]
    .value_counts()
    .head(20)
)

print(
    "\nTotal TLD categories after grouping:",
    df["TLD_grouped"].nunique()
)


# ============================================================
# 5. Check OTHER category
# ============================================================

other_count = (df["TLD_grouped"] == "OTHER").sum()

other_percentage = other_count / len(df) * 100

print("\n--- OTHER Category ---")
print("Number of URLs:", other_count)
print("Percentage:", round(other_percentage, 2), "%")


# ============================================================
# 6. One-hot encode grouped TLD
# ============================================================

tld_encoded = pd.get_dummies(
    df["TLD_grouped"],
    prefix="TLD",
    dtype=int
)


# ============================================================
# 7. Display encoding information
# ============================================================

print("\n--- One-Hot Encoding ---")

print(
    "Encoded TLD feature count:",
    tld_encoded.shape[1]
)

print("\nEncoded columns:")
print(tld_encoded.columns.tolist())


# ============================================================
# 8. Combine encoded features with dataset
# ============================================================

df_encoded = pd.concat(
    [
        df.drop(
            columns=["TLD", "TLD_grouped"]
        ),
        tld_encoded
    ],
    axis=1
)


# ============================================================
# 9. Drop high-cardinality raw text features
# ============================================================

text_features_to_remove = [
    "URL",
    "Domain",
    "Title"
]

df_encoded = df_encoded.drop(
    columns=text_features_to_remove
)


# ============================================================
# 10. Final check
# ============================================================

print("\n--- Final Dataset After Encoding ---")

print("Shape:", df_encoded.shape)

print("\nRemaining object columns:")
print(
    df_encoded.select_dtypes(
        include=["object"]
    ).columns.tolist()
)


# ============================================================
# 11. SAVE ENCODED DATASET
# ============================================================

output_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

print("\n--- Saving Dataset ---")

df_encoded.to_csv(
    output_path,
    index=False
)

print("Encoded dataset saved successfully!")
print("Output file:", output_path)
