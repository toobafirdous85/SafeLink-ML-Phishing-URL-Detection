import pandas as pd

file_path = "data/phiusiil+phishing+url+dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

print("\n--- Highly Correlated Feature Pairs ---")

numeric_features = df.select_dtypes(include="number").drop(columns=["label"])

corr_matrix = numeric_features.corr()

high_corr = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        correlation_value = corr_matrix.iloc[i, j]

        if abs(correlation_value) >= 0.85:
            high_corr.append(
                (
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    correlation_value
                )
            )

high_corr = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)

for feature1, feature2, correlation_value in high_corr:
    print(f"{feature1} <-> {feature2}: {correlation_value:.3f}")
