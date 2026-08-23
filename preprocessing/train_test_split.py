import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load the encoded dataset
# --------------------------------------------------

file_path = "data/phiusiil+phishing+url+dataset/encoded_dataset.csv"

df = pd.read_csv(file_path)

print("Encoded dataset loaded successfully!")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["label"])
y = df["label"]

print("\n--- Feature / Target Separation ---")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget distribution (%):")
print(y.value_counts(normalize=True) * 100)


# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

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

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# --------------------------------------------------
# 4. Verify target distribution
# --------------------------------------------------

print("\n--- Training Target Distribution (%) ---")
print(y_train.value_counts(normalize=True) * 100)

print("\n--- Testing Target Distribution (%) ---")
print(y_test.value_counts(normalize=True) * 100)


# --------------------------------------------------
# 5. Final validation
# --------------------------------------------------

print("\n--- Final Validation ---")

print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())

print("Object columns in X_train:", X_train.select_dtypes(include="object").columns.tolist())
print("Object columns in X_test:", X_test.select_dtypes(include="object").columns.tolist())

print("Label present in X_train:", "label" in X_train.columns)
print("Label present in X_test:", "label" in X_test.columns)

print("\nPreprocessing pipeline validation completed successfully!")
