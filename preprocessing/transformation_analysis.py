import pandas as pd

file_path = "data/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Remove duplicate URLs
df = df.drop_duplicates(subset="URL", keep="first")

# Remove unnecessary/redundant features
df = df.drop(columns=[
    "FILENAME",
    "NoOfLettersInURL",
    "URLTitleMatchScore"
])

# Count-based features
count_features = [
    "NoOfSubDomain",
    "NoOfObfuscatedChar",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "LineOfCode",
    "LargestLineLength",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef"
]

# Calculate skewness
skewness = df[count_features].skew().sort_values(ascending=False)

print("\n--- Count Feature Skewness ---")
print(skewness)

# Select positively skewed count features
log_features = skewness[skewness > 1].index.tolist()

print("\n--- Features Selected For log1p Transformation ---")

for feature in log_features:
    print(feature)

print("\nNumber of features selected:", len(log_features))
