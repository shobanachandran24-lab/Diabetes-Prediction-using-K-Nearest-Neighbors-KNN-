# Diabetes Prediction using K-Nearest Neighbors (KNN)

## Project Description

This project predicts whether a person has diabetes based on medical measurements. It uses the **Pima Indians Diabetes dataset** (`diabetes.csv`) and builds a **K-Nearest Neighbors (KNN) classifier** in a Jupyter Notebook (`KNN_Regression.ipynb`). A **Logistic Regression** model is then built on the same data as a second model.

The target is binary: **1 = the person has diabetes, 0 = the person does not**.

## Files in the Project

| File | Purpose |
|---|---|
| `diabetes.csv` | The dataset: 768 patient records, 9 columns |
| `KNN_Regression.ipynb` | The notebook with all the code: data checking, splitting, KNN, evaluation and Logistic Regression |
| `README.md` | This documentation |

## About the Data

The dataset has **768 rows** (patients) and **9 columns**: 8 input features and 1 target column. All values are numeric. There are **no missing values** and **no duplicate rows**.

### Columns

| Column | Type | Meaning | Min | Mean | Max |
|---|---|---|---|---|---|
| `Pregnancies` | int | Number of times pregnant | 0 | 3.85 | 17 |
| `Glucose` | int | Plasma glucose concentration | 0 | 120.89 | 199 |
| `BloodPressure` | int | Diastolic blood pressure (mm Hg) | 0 | 69.11 | 122 |
| `SkinThickness` | int | Triceps skin-fold thickness (mm) | 0 | 20.54 | 99 |
| `Insulin` | int | 2-hour serum insulin (mu U/ml) | 0 | 79.80 | 846 |
| `BMI` | float | Body mass index | 0 | 31.99 | 67.1 |
| `DiabetesPedigreeFunction` | float | Score of diabetes family history | 0.08 | 0.47 | 2.42 |
| `Age` | int | Age in years | 21 | 33.24 | 81 |
| `Outcome` | int | **Target**: 0 = no diabetes, 1 = diabetes | 0 | 0.35 | 1 |

### Target distribution

| Outcome | Count | Share |
|---|---|---|
| 0 (no diabetes) | 500 | 65.1% |
| 1 (diabetes) | 268 | 34.9% |

## What Was Done in the Project (Step by Step)

### 1. Import libraries
NumPy, Pandas, Matplotlib and Seaborn were imported, and warnings were turned off.

### 2. Load the data
`diabetes.csv` was read into a Pandas DataFrame called `df`.

### 3. Understand and check the data
The following checks were run:
- `df.head()` to view the first rows
- `df.shape` to get the size (768 rows, 9 columns)
- `df.info()` and `df.dtypes` to see the column types
- `df.isnull().sum()` to count missing values (result: none)
- `df.duplicated().sum()` to count duplicate rows (result: none)

### 4. Define features and target
- **Features (x):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target (y):** Outcome

### 5. Split into training and testing sets
`train_test_split` was used with `test_size=0.2` and `random_state=42`:
- Training set: 614 rows (80%)
- Testing set: 154 rows (20%), of which 99 are non-diabetic and 55 are diabetic

### 6. Find a suitable value of k
A KNN classifier was trained for every k from 1 to 10. For each k, the accuracy was calculated on both the training data and the testing data. The results were plotted twice as line graphs (accuracy against k).

| k | Train accuracy | Test accuracy |
|---|---|---|
| 1 | 1.000 | 0.675 |
| 2 | 0.832 | 0.701 |
| 3 | 0.855 | 0.649 |
| 4 | 0.809 | 0.708 |
| 5 | 0.798 | 0.662 |
| 6 | 0.788 | 0.727 |
| 7 | 0.793 | 0.688 |
| 8 | 0.777 | 0.747 |
| 9 | 0.793 | 0.721 |
| 10 | 0.790 | 0.766 |

### 7. Build the final KNN model
The final model is `KNeighborsClassifier(n_neighbors=6)` (the scikit-learn default is 5, and this project uses 6). It was trained on the training set.

### 8. Make predictions
The model predicted the Outcome for the test set, and the first five predictions were compared with the first five actual values.

### 9. Evaluate the KNN model
The test set predictions were evaluated with accuracy, precision, recall, F1-score, a confusion matrix (drawn as a Seaborn heatmap) and a classification report.

| Metric | Value |
|---|---|
| Accuracy | 0.727 |
| Precision | 0.633 |
| Recall | 0.564 |
| F1-score | 0.596 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 81 | 18 |
| **Actual 1** | 24 | 31 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (no diabetes) | 0.77 | 0.82 | 0.79 | 99 |
| 1 (diabetes) | 0.63 | 0.56 | 0.60 | 55 |
| Accuracy | | | 0.73 | 154 |

### 10. Model 2: Logistic Regression
A `LogisticRegression` model was trained on the same training data and tested on the same test data. Its score, confusion matrix, precision and classification report were produced.

| Metric | Value |
|---|---|
| Accuracy | 0.747 |
| Precision | 0.638 |
| Recall | 0.673 |
| F1-score | 0.655 |

Confusion matrix (rows = actual, columns = predicted):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 78 | 21 |
| **Actual 1** | 18 | 37 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (no diabetes) | 0.81 | 0.79 | 0.80 | 99 |
| 1 (diabetes) | 0.64 | 0.67 | 0.65 | 55 |
| Accuracy | | | 0.75 | 154 |

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| KNN (k = 6) | 0.727 | 0.633 | 0.564 | 0.596 |
| Logistic Regression | 0.747 | 0.638 | 0.673 | 0.655 |

Both models were tested on the same 154 test samples. Logistic Regression scored slightly higher on all four metrics.

## Technologies Used

- Python 3
- NumPy and Pandas (data handling)
- Matplotlib and Seaborn (graphs and heatmap)
- Scikit-learn (train/test split, KNN, Logistic Regression, metrics)
- Jupyter Notebook

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook KNN_Regression.ipynb
```

Keep `diabetes.csv` in the same folder as the notebook.

## Dataset Source

Pima Indians Diabetes Database, from the National Institute of Diabetes and Digestive and Kidney Diseases (available on Kaggle and the UCI Machine Learning Repository).
