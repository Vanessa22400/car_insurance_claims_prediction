# Car Insurance Claim Prediction — Single Feature Logistic Models

### Project Overview

Insurance companies need to estimate risk correctly in order to price their products well.

In this project, the insurance company **On the Road Insurance** asked for a simple machine learning solution to help them identify which single feature is the best predictor of whether a customer will make a claim during the policy period.

Because the company does not have much experience with machine learning or deployment, **the goal was**:
> to build one logistic regression model for each feature, compare the results, and select the feature that gives the highest accuracy.

This repository includes the data analysis, data cleaning steps, model creation, and final results.

### Dataset

The dataset contains 10,000 customers and several features about:

* personal information
* driving history
* vehicle information
* past incidents

The target variable is:

* `outcome`: 1 if the customer made a claim, 0 otherwise.

Examples of features in the dataset:

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
* and others.

The column `id` was removed because it does not help with prediction.


### Data Preparation

The following steps were performed before modeling:

* Missing values in `credit_score` and `annual_mileage` were replaced with the mean of each column.
* All other variables were kept as they were.
* A logistic regression model was trained for each feature separately.


### Modeling Approach

The goal was to see which individual feature predicts the target variable the best.

For each feature, a logistic regression was fitted using the formula:

```python
outcome ~ feature
```

For every model, accuracy was calculated using the confusion matrix.
The feature with the highest accuracy was selected as the best predictor.


### Results

* the feature that produced the best accuracy was `driving_experience`
* with an accuracy of **0.7771**

This result makes sense, because drivers with less experience often have a higher chance of being involved in accidents, which increases the likelihood of making a claim.



### How to Run the Code

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# Load dataset
car = pd.read_csv('car_insurance.csv')

# Replace missing values
car['credit_score'].fillna(car['credit_score'].mean(), inplace=True)
car['annual_mileage'].fillna(car['annual_mileage'].mean(), inplace=True)

# Select features
features_cols = car.columns.drop(["outcome", "id"])

# Train logistic models
models = []
for col in features_cols:
    model = logit(f"outcome ~ {col}", data=car).fit(disp=0)
    models.append(model)

# Compute accuracy
accuracy = []
for i in range(len(models)):
    table = models[i].pred_table()
    tn, fp, fn, tp = table.ravel()
    accuracy.append((tp + tn) / (tp + tn + fp + fn))

# Best feature
best_feature = features_cols[np.argmax(accuracy)]
best_accuracy = max(accuracy)

print(best_feature, best_accuracy)
```



### Conclusion

This project shows a simple method to find the most important single feature for predicting car insurance claims.
By training 1 logistic model per feature, we identified **'driving_experience' as the strongest predictor**.

> This approach is useful for companies that are just starting to use machine learning and want a model that is easy to understand and deploy.


### Possible Future Improvements

Some possible next steps include:
* Using multiple features in the same model
* Testing other algorithms such as Random Forest or Gradient Boosting
* Applying cross-validation
* Analyzing feature importance
* Building a small API for deployment
