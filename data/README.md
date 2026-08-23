# Dataset

## Dataset Name

**PhiUSIIL Phishing URL Dataset**

The SafeLink project uses the **PhiUSIIL Phishing URL Dataset**, which contains URL-level features designed for phishing URL classification.

## Local Dataset Structure

For local execution, the dataset is expected to be placed in the following directory:

```text
data/
└── phiusiil+phishing+url+dataset/
    └── PhiUSIIL_Phishing_URL_Dataset.csv
```

The preprocessing scripts in the `preprocessing/` directory use this path when loading the dataset.

## Dataset Usage

The dataset is used throughout the Week 2 preprocessing pipeline for:

* Dataset inspection
* Data quality assessment
* Duplicate URL analysis
* Feature selection
* Numerical feature transformation
* Scaling analysis
* Categorical feature analysis
* TLD encoding
* Train-test splitting

## Generated Files

During preprocessing, intermediate datasets may be generated locally, including:

```text
transformed_dataset.csv
encoded_dataset.csv
```

These generated datasets are used as intermediate stages of the preprocessing pipeline.

## Repository Policy

The raw dataset and generated datasets are not committed to this repository. This keeps the repository lightweight while allowing the preprocessing implementation and methodology to remain fully documented and reproducible.

To execute the preprocessing scripts locally, place the dataset in the directory structure described above.

## Expected Dataset Dimensions

The original dataset used during Week 2 contained:

* **235,795 records**
* **56 columns**

After duplicate URL removal and preprocessing, the final machine-learning-ready representation contained:

* **235,370 records**
* **100 columns**
