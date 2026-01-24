# House Prices – Machine Learning Project

## Project Goal
The goal of this project is to predict house sale prices using machine learning.

The dataset comes from the Kaggle *House Prices* competition and contains different characteristics of houses such as size, location, and available equipment.

---

## Dataset Description
Each row represents a house, and each column represents a feature describing the house (land size, neighborhood, garage, basement, etc.).

The target variable of the project is **SalePrice**, which is the final price at which the house was sold.

The dataset is not included in this repository. It can be downloaded from Kaggle and placed in a `data/` folder.

---

## Project Steps

### 1. Data Exploration
First, the dataset was explored to understand its structure and the meaning of each feature.  
This step helped identify numerical and categorical variables, as well as missing values.

---

### 2. Data Cleaning
Missing values were handled differently depending on their meaning:
- Missing values representing the absence of an equipment (garage, pool, basement, etc.) were replaced by `"None"`
- Numerical missing values were replaced using simple and robust strategies such as the median or zero when it made sense

This step was necessary to make the dataset usable by machine learning models.

---

### 3. Feature Preparation
The dataset was split into:
- numerical features
- categorical features

Categorical variables were encoded using **One-Hot Encoding** so that they could be used by the model.

A preprocessing pipeline was used to apply the same transformations automatically.

---

### 4. Model Training
The dataset was split into training and validation sets.

A **Random Forest Regressor** was trained to predict house prices based on the available features.  
The model was integrated into a pipeline with the preprocessing steps.

---

### 5. Model Evaluation
The model was evaluated using **Mean Absolute Error (MAE)**.

The final MAE is around **$17,700**, which means that, on average, the predicted price differs from the real price by about $17,700.

---

## Results and Observations
- The model performs well on standard houses
- Larger prediction errors mainly occur for very expensive or atypical houses
- Visual comparisons between predicted and real prices were used to better understand the model’s behavior

---

## Future Improvements
Several improvements could be explored in future work:
- Applying a logarithmic transformation to the target variable to reduce the impact of extreme values
- Testing more advanced models such as Gradient Boosting or XGBoost
- Performing hyperparameter tuning to improve model performance
- Adding feature engineering to better capture house quality and interactions between variables
- Using cross-validation instead of a single train/validation split

---

## Tools Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Conclusion
This project allowed me to apply a complete machine learning workflow, from data understanding and cleaning to model training and evaluation.  
It also helped me better understand the importance of data preprocessing and model evaluation in real-world machine learning projects.
