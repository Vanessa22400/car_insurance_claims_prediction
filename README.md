# Car Insurance Claim Prediction 

### Project Overview

Insurance companies need to estimate risk correctly in order to price their products well and prevent financial losses.

In this project, I built 2 baseline machine learning models to predict whether a customer will file an insurance claim during the policy period of an insurance company. 

The final objective was:

**Compare Logistic Regression and Random Forest to identify which model performs better for this classification task.**

This repository includes the full pipeline: exploration, preprocessing, encoding, model training, and evaluation.

---

## Dataset

The dataset contains 10,000 customers and several features about:

* personal information
* driving history
* vehicle information
* past incidents

### Target Variable

* `outcome`
    * 1 if the customer made a claim
    * 0 if the customer do not made a claim

### Example Features

* age
* gender
* driving_experience
* education
* income
* credit_score
* annual_mileage
* vehicle_type
* speeding_violations
* past_accidents
* married  
* children  
* vehicle_year 

The column `id` was removed because it does not help with prediction.

---

## Data Preparation

The following steps were performed before modeling:

1. **Handling Missing Values**  
   - Missing values in **credit_score** and **annual_mileage** were replaced using the **mean**.

2. **Encoding Categorical Variables**  
   - Used `pandas.get_dummies()` to convert categorical variables into numerical format.

3. **Train-Test Split**  
   - 80% training  
   - 20% testing  

4. **Feature Selection**  
   - All columns (except `id` and `outcome`) were used as inputs.

---

### Modeling Approach

The goal was to see which individual feature predicts the target variable the best.

For each feature, a logistic regression was fitted using the formula:

```python
outcome ~ feature
```

For every model, accuracy was calculated using the confusion matrix.
The feature with the highest accuracy was selected as the best predictor.


## Results

### Accuracy

| Model                | Accuracy |
|---------------------|-----------|
| Logistic Regression | **0.803** |
| Random Forest       | **0.823** |

### Interpretation

#### Logistic Regression
- Strong performance for class **0** (no claim).  
- Lower recall for class **1**, meaning it **misses some claim cases**.

#### Random Forest
- Best model overall, with higher recall for class **1**, which is important for not missing potential claims.  
- More balanced results across all metrics.

### Conclusion

**Random Forest outperformed Logistic Regression**, achieving the highest accuracy and capturing more true claim cases.  
Logistic Regression remains a useful and interpretable baseline.

---

### How to Run the Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
car = pd.read_csv("car_insurance.csv")

# Handle missing values
car["credit_score"].fillna(car["credit_score"].mean(), inplace=True)
car["annual_mileage"].fillna(car["annual_mileage"].mean(), inplace=True)

# Remove id column
car = car.drop("id", axis=1)

# Encode categorical variables
car_encoded = pd.get_dummies(car, drop_first=True)

# Train-test split
X = car_encoded.drop("outcome", axis=1)
y = car_encoded["outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, solver="liblinear")
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
log_preds = log_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Metrics
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

print(confusion_matrix(y_test, log_preds))
print(confusion_matrix(y_test, rf_preds))

print(classification_report(y_test, log_preds))
print(classification_report(y_test, rf_preds))
```
