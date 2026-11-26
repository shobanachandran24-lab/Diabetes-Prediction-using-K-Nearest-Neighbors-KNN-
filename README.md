# Diabetes Prediction using K-Nearest Neighbors (KNN)

> A machine learning project that predicts whether a patient has diabetes from diagnostic measurements, using a KNN classifier and comparing it with Logistic Regression.

## Description

This project uses the **Pima Indians Diabetes dataset** to predict the onset of diabetes (`Outcome`: 1 = diabetic, 0 = non-diabetic) from eight medical features such as glucose level, BMI, age and insulin.

The notebook walks through a complete beginner-friendly ML workflow:

1. Loading and inspecting the data (shape, types, missing values, duplicates)
2. Selecting features and the target variable
3. Splitting the data into training (80%) and testing (20%) sets
4. Tuning `k` for KNN by comparing train vs. test accuracy for k = 1 to 10
5. Building the final KNN model (k = 6) and making predictions
6. Evaluating with accuracy, precision, recall, F1-score, a confusion matrix and a classification report
7. Training a second model (Logistic Regression) for comparison

> **Note:** Although the notebook file is named `KNN_Regression.ipynb`, the task is **binary classification** (the target is 0/1), so it uses `KNeighborsClassifier`, not a regression model.

## Project Structure

```
.
├── KNN_Regression.ipynb   # Main notebook (analysis, modeling, evaluation)
├── diabetes.csv           # Dataset (768 rows, 9 columns)
└── README.md
```

## Dataset

| Property | Value |
|---|---|
| Rows | 768 patients |
| Features | 8 numeric columns |
| Target | `Outcome` (0 = no diabetes, 1 = diabetes) |
| Class balance | 500 non-diabetic (65.1%), 268 diabetic (34.9%) |
| Missing values / duplicates | None |

**Features**

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes family-history score |
| `Age` | Age in years |

**Data quality note:** several columns contain `0` where a zero is not physiologically possible, which usually means a missing value (Glucose: 5, BloodPressure: 35, SkinThickness: 227, Insulin: 374, BMI: 11). The current notebook does not treat these, which is a good area for improvement (see below).

## Technologies Used

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook KNN_Regression.ipynb
   ```
   Make sure `diabetes.csv` is in the same folder as the notebook.

## Models and Results

Both models were evaluated on the same 20% hold-out test set (154 samples, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| KNN (k = 6) | 0.727 | 0.633 | 0.564 | 0.596 |
| Logistic Regression | 0.747 | 0.638 | 0.673 | 0.655 |

Logistic Regression performs slightly better overall, mainly with higher recall, meaning it misses fewer diabetic patients. In a medical screening setting, recall is usually the metric that matters most.

**Choosing k:** with k = 1 the model memorizes the training data (100% train accuracy, about 67.5% test accuracy), which is classic overfitting. Test accuracy changes noticeably from one k to the next, so a more robust choice of `k` would use cross-validation.

## Possible Improvements

- **Scale the features** (e.g. `StandardScaler`). KNN is distance-based, so features with large ranges like Insulin and Glucose dominate the distance calculation.
- **Handle the invalid zeros** by replacing them with the median or using imputation.
- **Use cross-validation and `GridSearchCV`** to choose `k`, distance metric and weighting instead of a single train/test split.
- **Address class imbalance** with stratified splitting (`stratify=y`) or class weights.
- Try other models such as Random Forest, SVM or Gradient Boosting.
- Report ROC-AUC alongside the other metrics.

## Disclaimer

This project is for learning and demonstration purposes only. It is not a medical tool and must not be used for real diagnosis or treatment decisions.

## Acknowledgements

Dataset: Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases (available on Kaggle and the UCI Machine Learning Repository).

## License

Add your preferred license here (for example, MIT).
