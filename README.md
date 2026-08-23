# SafeLink – ML-Based Phishing URL Detection

## Project Overview

**SafeLink** is a machine learning-based cybersecurity project designed to analyze URL characteristics and classify URLs as **legitimate or potentially phishing**.

The project follows an end-to-end machine learning workflow, beginning with dataset analysis and preprocessing, followed by feature engineering, model development, evaluation, and the eventual development of a practical URL prediction system.

The primary objective is to develop a **reliable, explainable, and practically deployable phishing URL detection pipeline** based on URL-level characteristics.

---

## Problem Statement

Phishing attacks frequently rely on deceptive URLs to direct users toward malicious websites and compromise sensitive information. Because phishing URLs can be designed to closely resemble legitimate URLs, manual identification can be difficult and unreliable.

SafeLink addresses this problem by developing a machine learning system that analyzes measurable URL characteristics and classifies URLs into two categories:

* **Legitimate**
* **Phishing / Potentially Malicious**

The system focuses on URL-based characteristics rather than depending exclusively on webpage content or visual appearance.

---

## Project Objectives

The major objectives of SafeLink are to:

1. Understand URL characteristics commonly associated with phishing activity.
2. Identify and prepare a suitable dataset containing legitimate and phishing URLs.
3. Perform systematic data inspection, cleaning, and preprocessing.
4. Engineer and transform meaningful URL-based features.
5. Prepare a machine-learning-ready feature matrix.
6. Develop and compare suitable classification algorithms.
7. Evaluate model performance using appropriate classification metrics.
8. Analyze model errors and improve predictive performance.
9. Develop a practical workflow for classifying previously unseen URLs.
10. Document the complete machine learning development process.

---

# Machine Learning Pipeline

SafeLink follows an end-to-end machine learning workflow:

**Problem Definition**
↓
**Dataset Collection & Understanding**
↓
**Data Inspection & Quality Assessment**
↓
**Data Cleaning**
↓
**Feature Selection & Engineering**
↓
**Numerical Transformation**
↓
**Categorical Encoding**
↓
**Feature Scaling Analysis**
↓
**Train-Test Split**
↓
**Model Development**
↓
**Model Evaluation**
↓
**Model Comparison & Improvement**
↓
**Final Prediction System**

---

# Dataset

The project uses a publicly available dataset containing URL-level features for phishing URL classification.

The original dataset contained:

* **235,795 records**
* **56 columns**

The dataset includes URL-derived characteristics representing structural, lexical, and other measurable properties of URLs.

The dataset was systematically inspected during Week 2 to identify missing values, duplicate records, duplicate URLs, inconsistent values, redundant features, skewed numerical distributions, and categorical variables.

> **Note:** The raw dataset is not included directly in this repository due to its size. The preprocessing methodology and implementation are documented within the repository.

---

# Week 1 – Project Planning & Strategy

**Status: Completed**

Week 1 established the foundation of the SafeLink project.

### Activities Completed

* Defined the phishing URL detection problem.
* Established project objectives and scope.
* Designed the overall machine learning pipeline.
* Identified relevant URL-based feature categories.
* Considered suitable machine learning algorithms.
* Defined appropriate evaluation metrics.
* Established the six-week development roadmap.
* Set up the initial GitHub repository and project documentation.

Week 1 focused primarily on **problem definition, strategy, planning, and project architecture**.

---

# Week 2 – Data Preprocessing & Feature Engineering

**Status: Completed**

Week 2 focused on transforming the raw dataset into a clean, consistent, and machine-learning-ready representation.

The preprocessing stage was implemented systematically rather than relying solely on exploratory inspection.

## Dataset Inspection

The original dataset contained:

**235,795 rows × 56 columns**

A detailed inspection was performed to understand:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Duplicate URLs
* Target distribution
* Numerical feature distributions
* Categorical variables
* Potentially redundant features
* Feature skewness

## Data Cleaning

The dataset was examined for missing and inconsistent values.

### Missing Values

No missing values requiring imputation were identified in the dataset.

### Duplicate Records

Duplicate records were investigated as part of the data-quality assessment.

### Duplicate URLs

Duplicate URL entries were identified and removed to reduce redundancy and prevent repeated URLs from unnecessarily influencing downstream analysis.

**425 duplicate URL records were removed.**

### Conflicting Labels

Potentially conflicting labels associated with repeated URLs were investigated.

**No conflicting labels were identified.**

---

## Feature Selection

Redundant or unnecessary variables were evaluated and removed where appropriate.

The following features were removed:

* `NoOfLettersInURL`
* `URLTitleMatchScore`
* `FILENAME`

Feature removal was based on redundancy, relevance, and suitability for the subsequent machine learning pipeline.

---

## Numerical Feature Transformation

Numerical feature distributions were analyzed to identify highly skewed variables.

Appropriate transformations were applied to **17 skewed count-based features** in order to reduce the effect of extreme values and improve feature distributions.

The transformations were incorporated into the preprocessing pipeline rather than being performed arbitrarily.

---

## Categorical Encoding

Categorical URL-related information, particularly the **TLD (Top-Level Domain)** feature, was converted into a machine-learning-compatible numerical representation.

The encoding strategy included an **OTHER category** to group less frequently represented TLD values and control excessive categorical dimensionality.

The final encoding contained:

* **51 TLD categories including OTHER**
* **19,249 records assigned to OTHER**
* **OTHER category proportion: 8.18%**

After preprocessing, no remaining object-type columns were present in the final dataset.

---

## Final Dataset

After completing the cleaning, feature selection, transformation, and encoding stages, the resulting dataset contained:

| Stage                       |        Rows | Columns |
| --------------------------- | ----------: | ------: |
| Original dataset            |     235,795 |      56 |
| After duplicate URL removal |     235,370 |      56 |
| Final preprocessed dataset  | **235,370** | **100** |

The increase in dimensionality is primarily associated with categorical encoding, while the reduction in rows resulted from duplicate URL removal.

### Final preprocessing outcome

**235,370 rows × 100 features**

The resulting dataset is prepared for the subsequent machine learning development stage.

---

# Week 3 – Model Development

**Status: Planned**

The next stage will focus on developing baseline machine learning classification models.

Planned activities include:

* Train-test data preparation.
* Baseline model development.
* Training multiple classification algorithms.
* Establishing initial performance benchmarks.
* Comparing model behaviour and computational requirements.

Potential models include:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Naive Bayes

---

# Week 4 – Model Evaluation & Improvement

**Status: Planned**

This stage will focus on systematic model evaluation and improvement.

Planned activities include:

* Accuracy analysis.
* Precision and recall analysis.
* F1-score comparison.
* Confusion matrix analysis.
* Error and misclassification analysis.
* Hyperparameter tuning.
* Model comparison.
* Selection of the most suitable model.

Particular attention will be given to **false negatives**, where phishing URLs may incorrectly be classified as legitimate.

---

# Week 5 – Prediction System & Integration

**Status: Planned**

The selected model will be integrated into a practical prediction workflow.

Planned activities include:

* Accepting previously unseen URLs.
* Applying the same preprocessing pipeline.
* Generating model predictions.
* Producing interpretable prediction outputs.
* Testing the system on representative URLs.
* Improving usability and reliability.

---

# Week 6 – Finalization & Documentation

**Status: Planned**

The final stage will consolidate the completed project.

Planned activities include:

* Final system testing.
* Performance analysis.
* Repository organization.
* Documentation completion.
* Final results presentation.
* Preparation of the completed SafeLink project.

---

# Technologies

| Category                | Technology                |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Data Processing         | Pandas, NumPy             |
| Machine Learning        | Scikit-learn              |
| Data Visualization      | Matplotlib                |
| Development Environment | Jupyter Notebook / Python |
| Version Control         | Git & GitHub              |

Additional libraries may be introduced as the project progresses.

---

# Repository Structure

```text
SafeLink-ML-Phishing-URL-Detection/
│
├── README.md
│
├── requirements.txt
│
├── preprocessing/
│   ├── README.md
│   ├── inspect_dataset.py
│   ├── cleaning.py
│   ├── feature_selection.py
│   ├── transformation_analysis.py
│   ├── transformation.py
│   ├── scaling_analysis.py
│   └── categorical_encoding.py
│
├── outputs/
│   └── preprocessing analysis and visualizations
│
└── reports/
    └── Week_2_Report.pdf
```

The repository structure will evolve as additional stages of the project are implemented.

---

# Evaluation Metrics

The machine learning models will be evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

Because phishing detection is a security-sensitive classification problem, particular importance will be placed on **Recall and false-negative analysis**, while maintaining an appropriate balance with Precision.

---

# Future Scope

SafeLink can be extended through:

* Additional URL and domain-based features.
* Advanced feature engineering.
* Ensemble learning approaches.
* Hyperparameter optimization.
* Robustness testing against evolving phishing techniques.
* Real-time URL analysis.
* Web-based prediction interfaces.
* Browser-based security integration.
* Explainable AI techniques for prediction interpretation.

---

# Project Information

**Project Name:** SafeLink – ML-Based Phishing URL Detection
**Domain:** Machine Learning & Cybersecurity
**Development Approach:** End-to-End Machine Learning Pipeline
**Current Phase:** **Week 2 – Data Preprocessing & Feature Engineering**
**Project Duration:** Six Weeks

---

# Repository Status

The SafeLink repository is being developed incrementally throughout the internship.

**Current status: Week 2 completed.**

The project has progressed from initial planning and strategy development to the implementation of a structured data preprocessing and feature engineering pipeline.

The resulting dataset contains **235,370 records and 100 machine-learning-ready features**, providing the foundation for the upcoming model development stage.

---

## Documentation

Detailed weekly implementation reports will be maintained in the `reports/` directory.

* **Week 1:** Project Planning & Strategy
* **Week 2:** Data Preprocessing & Feature Engineering
* **Week 3:** Model Development *(upcoming)*
* **Week 4:** Model Evaluation & Improvement *(upcoming)*
* **Week 5:** Prediction System & Integration *(upcoming)*
* **Week 6:** Finalization & Documentation *(upcoming)*

---

## Note

SafeLink is being developed as a structured six-week machine learning internship project. The repository will be updated progressively as each stage of the machine learning pipeline is implemented, evaluated, and documented.


