# Module 2 – Analytics Pipeline

## Overview

This module demonstrates a complete analytics workflow using the Titanic dataset. The workflow includes dataset profiling, data cleaning, exploratory data analysis (EDA), feature preprocessing, predictive modeling, model evaluation, hyperparameter tuning, regression analysis, and deployment of the final machine learning pipeline.

The Titanic dataset was loaded only once using Seaborn's built-in dataset loader and immediately saved as `titanic.csv` as an offline fallback. All subsequent analysis and modeling were performed using this saved dataset.

---

# Dataset

- **Source:** Seaborn Titanic Dataset
- **Records:** 891
- **Features:** 15
- **Target Variable:** `survived`

---

# Part A – Exploratory Data Analysis

## Dataset Profiling

The dataset was profiled using:

- `df.info()`
- `df.describe()`
- `df.shape`

These methods were used to inspect data types, identify missing values, and understand the statistical properties of the dataset before preprocessing.

---

## Missing Value Handling

The following threshold rule was applied:

- **Missing values below 5%** → Drop affected rows
- **Missing values between 5% and 30%** → Impute missing values
- **Missing values above 30%** → Remove the column

| Column | Missing % | Strategy | Justification |
|---------|----------:|----------|---------------|
| Age | **19.87%** | Median Imputation | Missing percentage falls between 5% and 30%, therefore median imputation was used to preserve data while reducing the influence of outliers. |
| Embarked | **0.22%** | Drop Missing Rows | Missing percentage is below 5%, therefore the affected rows were removed. |
| Deck | **77.22%** | Column Removed | Missing percentage exceeds 30%, making reliable imputation impractical. |

---

## Univariate Analysis

### Age

- Histogram and box plot were created.
- Outliers were detected using the IQR method.
- **Total Age Outliers:** **65**

### Fare

- Histogram and box plot were created.
- Outliers were detected using the IQR method.
- **Total Fare Outliers:** **114**

### Fare Distribution

The mean, median, and mode of Fare were computed.

Since:

**Mean > Median > Mode**

the Fare distribution is **positively (right) skewed**, indicating that a small number of passengers paid significantly higher fares than the majority.

---

# Bivariate Analysis

## Survival by Gender

Female passengers exhibited a significantly higher survival rate than male passengers, indicating that gender played an important role in survival.

## Survival by Passenger Class

Passengers travelling in first class experienced the highest survival rate, while third-class passengers had the lowest survival rate, showing that passenger class strongly influenced survival.

## Survival by Gender and Passenger Class

Females travelling in first class had the highest survival probability, whereas males travelling in third class had the lowest survival probability. This demonstrates that both gender and socioeconomic status affected survival.

---

# Correlation Analysis

The correlation matrix was computed using the following six numerical variables:

- survived
- pclass
- age
- sibsp
- parch
- fare

The derived boolean variables `adult_male` and `alone` were excluded because they are redundant features derived from other variables.

### Strongest Correlations

1. **Fare ↔ Passenger Class**

A strong negative correlation exists because passengers travelling in higher classes generally paid higher fares.

2. **SibSp ↔ Parch**

A positive correlation exists because passengers travelling with siblings often also travelled with parents or children.

---

# Multivariate Data Story

## Age vs Survival

Younger passengers generally exhibited slightly higher survival rates than older passengers, although age alone was not a decisive factor.

## Fare vs Survival

Passengers paying higher fares had a considerably greater likelihood of survival, reinforcing the relationship between passenger class and survival.

## Embarked vs Survival

Passengers embarking from Cherbourg showed comparatively higher survival rates, likely because a greater proportion of first-class passengers boarded there.

## Pair Plot

The pair plot highlights relationships among the numerical variables. Fare and passenger class display clearer separation between survivors and non-survivors than age, making them stronger predictors of survival.

---

# Standardization

Age and Fare were standardized using **StandardScaler**.

The transformed variables achieved approximately:

- Mean ≈ 0
- Standard Deviation ≈ 1

confirming successful standardization.

---

# Part B – Predictive Modeling

## Train-Test Split

A **stratified train-test split** was performed before any preprocessing.

Stratification preserves the class distribution of the target variable in both the training and testing datasets, resulting in a more reliable and unbiased evaluation.

---

## Preprocessing

The preprocessing pipeline was fitted **only on the training data** and included:

- Median Imputation
- Most Frequent Imputation
- One-Hot Encoding
- StandardScaler
- ColumnTransformer
- Pipeline

This prevents data leakage and ensures proper model evaluation.

---

# Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|------|---------:|----------:|-------:|---------:|----:|
| Logistic Regression | **0.8090** | **0.7833** | **0.6912** | **0.7344** | **0.8610** |
| Decision Tree | **0.7697** | **0.6901** | **0.7206** | **0.7050** | **0.7541** |
| Random Forest | **0.8202** | **0.7813** | **0.7353** | **0.7576** | **0.8179** |

---

# Imbalance Handling

Three approaches were evaluated:

- Baseline
- Class Weight (`class_weight='balanced'`)
- SMOTE

| Method | Precision | Recall | F1 Score |
|---------|----------:|-------:|---------:|
| Baseline | **0.7813** | **0.7353** | **0.7576** |
| Class Weight | **0.7391** | **0.7500** | **0.7445** |
| SMOTE | **0.7460** | **0.6912** | **0.7176** |

The baseline Random Forest model achieved the highest F1-score while maintaining the best balance between precision and recall. Although Class Weight slightly improved recall, it reduced the overall F1-score. SMOTE produced the lowest F1-score among the three approaches. Therefore, the baseline Random Forest model was selected as the preferred approach for this dataset.

---

# Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest classifier.

### Best Parameters
n_estimators = 200
max_depth = 10
max_features = None
### Out-of-Bag (OOB) Score

**<your OOB score>**

---

# Regression Results

A multivariate Linear Regression model was trained to predict passenger Fare.

| Model | MAE | RMSE | R² | Adjusted R² |
|------|----:|-----:|----:|------------:|
| Linear Regression | **<your MAE>** | **<your RMSE>** | **<your R²>** | **<your Adjusted R²>** |

---

# Residual Analysis

The residual plot shows an increasing spread of residuals as the predicted Fare increases. This indicates the presence of **heteroscedasticity**, suggesting that the variance of prediction errors is not constant across the prediction range.

---

# Final Recommendation

Among the three classification models, the **Random Forest classifier** achieved the strongest overall performance with an **accuracy of 82.02%**, **recall of 73.53%**, and the **highest F1-score of 75.76%**. Although Logistic Regression achieved the highest AUC (0.8610), Random Forest provided the best overall balance between precision and recall. Therefore, the Random Forest classifier is recommended as the final deployment model.

---

# Project Files

```
analytics/
│
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── best_random_forest_pipeline.joblib
├── README.md
└── figures/
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib