import pandas as pd

# Load dataset
file_path = "data/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Check missing values
# --------------------------------------------------

print("\n--- Missing Values ---")
missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

# --------------------------------------------------
# 2. Check duplicate rows
# --------------------------------------------------

print("\n--- Duplicate Rows ---")
print("Duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# 3. Check duplicate URLs
# --------------------------------------------------

print("\n--- Duplicate URLs ---")
print("Duplicate URLs:", df["URL"].duplicated().sum())

# --------------------------------------------------
# 4. Check conflicting labels for duplicate URLs
# --------------------------------------------------

url_label_counts = df.groupby("URL")["label"].nunique()

conflicting_urls = (url_label_counts > 1).sum()

print("\n--- Conflicting URL Labels ---")
print("URLs with different labels:", conflicting_urls)

# --------------------------------------------------
# 5. Remove complete duplicate rows
# --------------------------------------------------

df = df.drop_duplicates()

print("\nShape after removing duplicate rows:", df.shape)

# --------------------------------------------------
# 6. Remove duplicate URLs
# --------------------------------------------------

df = df.drop_duplicates(subset="URL", keep="first")

print("Shape after removing duplicate URLs:", df.shape)

# --------------------------------------------------
# 7. Check target distribution after cleaning
# --------------------------------------------------

print("\n--- Target Distribution After Cleaning ---")
print(df["label"].value_counts())

print("\n--- Target Distribution (%) ---")
print(df["label"].value_counts(normalize=True) * 100)

print("\nPreprocessing cleaning stage completed successfully!")
