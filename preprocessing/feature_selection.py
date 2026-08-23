import pandas as pd

# Load dataset
file_path = "data/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Remove duplicate URLs
# --------------------------------------------------

df = df.drop_duplicates(subset="URL", keep="first")

print("Shape after removing duplicate URLs:", df.shape)

# --------------------------------------------------
# 2. Remove unnecessary / redundant features
# --------------------------------------------------

features_to_remove = [
    "FILENAME",
    "NoOfLettersInURL",
    "URLTitleMatchScore"
]

df_selected = df.drop(columns=features_to_remove)

print("\n--- Removed Features ---")

for feature in features_to_remove:
    print(feature)

# --------------------------------------------------
# 3. Display remaining columns
# --------------------------------------------------

print("\n--- Remaining Columns ---")
print(df_selected.columns.tolist())

# --------------------------------------------------
# 4. Display final shape
# --------------------------------------------------

print("\n--- Final Shape After Feature Selection ---")
print(df_selected.shape)

print("\nFeature selection completed successfully!")
