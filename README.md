# Car Insurance Claim Prediction

This project uses machine learning to predict whether a car insurance customer is likely to file a claim.
It follows a simple end-to-end workflow: cleaning the dataset, exploring patterns, training models, and comparing their performance.

## 1. Project Overview

The goal is to build a predictive model that helps insurance companies identify customers with a higher probability of filing a claim.
The dataset contains demographic and vehicle-related information.

This project is meant to be **clear**, **simple**, and **practical**, focusing on essential steps used in real data science work.

## 2. Dataset

The dataset includes features such as:

* Age
* Gender
* Driving Experience
* Annual Mileage
* Vehicle Type
* Income Category
* Number of previous accidents

The target variable is:
* **Claim*  — whether the customer filed a claim (Yes/No)

## 3. Methods

The project includes the following steps:

**1. Data loading and inspection**

**2. Cleaning missing values**

**3. Encoding categorical variables**

**4. Exploratory analysis to understand distributions**

**5. Training two baseline models**
* Logistic Regression
* Random Forest
  
**6. Comparing metrics**
* Focus on **accuracy** and **recall**, since the goal is to detect more claim cases.

## 4. Models Tested

#### **Logistic Regression**

A simple and fast baseline model.
It gives a basic understanding of how well linear separation works for this problem.

#### **Random Forest**

A more flexible and powerful model that handles non-linear patterns and interactions between features.

## 5. Results

Both models performed reasonably well, but:
**Random Forest performed better overall.**
* Higher recall, meaning it caught more of the true claim cases
* Good accuracy, without losing too much performance on normal customers
Logistic Regression still serves as a useful baseline.

## 6. Main Conclusion

** Random Forest is the better model for this dataset**, offering a stronger balance between accuracy and recall.
It is more reliable for identifying customers who are likely to file a claim.

## 7. Future Improvements

Some ideas to make the model stronger:
* Try additional models (Gradient Boosting, XGBoost)
* Tune hyperparameters
* Feature engineering (grouping age ranges, income bins, etc.)
* Test balancing methods if the classes are uneven
