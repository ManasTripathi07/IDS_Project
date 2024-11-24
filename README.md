---

# IDS Project - Team Executioners

## Project Overview

This project focuses on applying **Machine Learning (ML) classification algorithms** to a dataset to derive meaningful inferences and explore various approaches for effective classification. 

The primary goal is to recognize black-and-white rectangular pixels representing the 26 English capital letters, distorted through 20 fonts. The dataset contains 20,000 unique samples. 

### Key Features:
- Preprocessing and analyzing a dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition).
- Implementation of multiple ML classifiers, including:
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machines (SVM)**
  - **Naive Bayes**
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
- Selection of the most optimal classifier based on metrics such as **accuracy** and **F-beta scores**.

---

## Table of Contents

1. [Objective](#objective)
2. [System Requirements](#system-requirements)
3. [Dataset Description](#dataset-description)
4. [Steps Undertaken](#steps-undertaken)
5. [Classification and Results](#classification-and-results)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## Objective

The project aims to explore ML classification algorithms to:
- Identify the best-performing classifier for letter recognition.
- Analyze and preprocess the dataset to optimize performance.
- Generate meaningful insights about the data.

---

## System Requirements

Ensure the following software/tools are installed:

- **Python 3**: Primary language for implementation.
- **Jupyter Notebook**: For running and testing code.
- **Libraries Used**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

> **Note**: Use [Anaconda](https://www.anaconda.com/products/distribution) for an all-in-one setup.

---

## Dataset Description

**Source**: [UCI ML Repository - Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)

**Specifications**:
- **Features**: 16 attributes representing statistical measures of pixel distributions.
- **Target**: 26 classes representing English alphabet letters.

---

## Steps Undertaken

1. **Dataset Preprocessing**:
   - Converted raw data to CSV format.
   - Applied **One-Hot Encoding** and **Label Encoding** for non-numeric data.
   - Checked for missing/null values and cleaned the data.

2. **Data Normalization**:
   - Used **Min-Max Scaling** for uniform scaling.
   - Evaluated and compared with **Standard Scaling**.

3. **Exploratory Data Analysis**:
   - Investigated class distributions and relationships between attributes.
   - Used scatter plots and feature importance analysis (e.g., `ExtraTreesClassifier`).

4. **Data Partitioning**:
   - Split data into training and testing sets (e.g., 80%-20% split).
   - Maintained consistent class distributions across partitions.

---

## Classification and Results

### Algorithms Implemented:
1. **Naive Bayes**
2. **Logistic Regression**
3. **Decision Tree**
4. **SVM (Support Vector Machines)**
5. **Random Forest**
6. **K-Nearest Neighbors (KNN)**

### Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F-beta Score**

### Key Findings:
- **KNN** emerged as the most effective classifier with:
  - Accuracy: **93.5%**
  - F-Beta Score: **93.54%**
- Min-Max Scaling significantly improved classifier performance.

---

## Conclusion

- **K-Nearest Neighbors (KNN)** was identified as the best-performing classifier.
- Normalization and preprocessing played a critical role in improving classification accuracy.
- The project demonstrates the effectiveness of ML algorithms for pattern recognition tasks.

---

## References

1. [UCI Machine Learning Repository - Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
2. [scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Matplotlib](https://matplotlib.org/)

---

Feel free to contribute, raise issues, or provide feedback to improve the project! 😊

Prepared By -
1. Manas Tripathi         (22UCC061)
2. Anurag Singh           (22UCS027)
3. Meet Dipak Agrawal     (22UCC125)
4. Tanmay Mishra          (22UCC109)

--- 


