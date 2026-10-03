# Employee Salary Classification Model

A machine learning project that predicts whether an employee's annual income is above or below $50K using demographic, employment, and compensation-related features. The solution combines robust data preprocessing, model evaluation, and an interactive Streamlit deployment to make the prediction workflow accessible for real-time and batch inference.

## Live Demo

Try the deployed app here:

https://employee-salary-classification-system.streamlit.app

## Project Overview

This project addresses a classic binary classification problem: predicting salary class based on employee attributes from the Adult Census Income dataset.

- Problem type: Binary classification
- Target variable: income
- Labels: >50K and <=50K
- Use case: Workforce analytics, compensation review, and HR decision support
- Deployment: Streamlit web application for individual and batch prediction

## Why This Project Matters

This project demonstrates the end-to-end lifecycle of a machine learning solution:

- Data acquisition and exploration
- Data cleaning and preprocessing
- Feature engineering and outlier treatment
- Model selection and evaluation
- Model serialization and deployment
- Real-world inference via user-friendly interface

It is a strong portfolio project because it blends business relevance with technical depth in a compact, deployable format.

## Dataset

The notebook uses the Adult Census Income dataset, which contains 48,842 records and a mix of numerical and categorical variables.

### Key features used for training

- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- gender
- hours-per-week
- native-country
- capital-gain
- capital-loss
- educational-num
- fnlwgt

### Data preparation performed in the notebook

- Replaced missing values encoded as `?` with category labels such as `Others` and `Not Listed`
- Removed inconsistent workclass values such as `Without-pay` and `Never-worked`
- Filtered unrealistic records for age range and education level
- Applied outlier handling using thresholds and capping logic
- Transformed `capital-gain` and `capital-loss` using `log1p`
- Dropped the redundant `education` field after using `educational-num`

## Modeling Approach

A preprocessing pipeline was built to handle mixed data types properly.

### Preprocessing pipeline

- Numerical columns: `StandardScaler()`
- Categorical columns: `OneHotEncoder(handle_unknown='ignore')`
- Combined using `ColumnTransformer`
- Final model wrapped in a `Pipeline` for clean training and prediction flow

### Final classifier

- Model: `XGBClassifier`
- Number of estimators: 200
- Learning rate: 0.1
- Maximum depth: 7
- Subsample: 0.8
- Colsample by tree: 0.8
- Evaluation metric: `logloss`
- Random state: 42

### Train/test split

- Split strategy: 80/20 train-test split
- Stratification: yes (`stratify=y`)
- Random seed: 42

## Model Evaluation

The notebook compares multiple machine learning models to choose the most accurate model.

### Accuracy comparison

| Model | Accuracy |
| --- | ---: |
| XGBoost | 0.8728 |
| Logistic Regression | 0.8487 |
| Random Forest | 0.8531 |
| Decision Tree | 0.8131 |

The selected XGBoost model achieved the best performance, with overall test accuracy around 0.8726, making it the final model used in the packaged application.

### Evaluation metrics used

- Accuracy
- Classification report
- Confusion matrix
- Precision, recall, and F1-score across salary classes

## Application Features

The project includes a full interactive prediction system:

### Single prediction

- User inputs employee attributes through a Streamlit sidebar
- Model predicts either `>50K` or `<=50K`
- Output is displayed in a clean UI with a prediction result

### Batch prediction

- Upload CSV file containing the same feature columns
- Run predictions on the entire dataset
- Preview results in a table
- Download the output predictions as a CSV file

This makes the project useful for both individual classification and operational data processing.

## Tech Stack

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

## Repository Structure

- `app.py` — Streamlit web application for prediction
- `Employee Salary Prediction Model.ipynb` — notebook containing exploration, modeling, and evaluation
- `xgboost_salary_model.pkl` — saved trained model pipeline
- `requirements.txt` — project dependencies

## Setup Instructions

Use Python 3.12. The saved model was serialized with scikit-learn 1.5.1, so
use the pinned dependencies below rather than a newer scikit-learn version.

### 1. Create a virtual environment

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux, replace the first command with `python3.12 -m venv .venv`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Dependency versions are pinned to keep the runtime compatible with the serialized model artifact.

### 3. Run the application

```bash
streamlit run app.py
```

## Usage

### Example workflow

1. Launch the app
2. Enter employee details such as age, workclass, occupation, education, gender, and hours worked
3. Click the prediction button
4. The model returns whether the employee is likely to earn above or below $50K

### Batch upload

Upload a CSV file with the expected columns:

- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- gender
- hours-per-week
- native-country
- capital-gain
- capital-loss
- educational-num
- fnlwgt

## Business Impact

This project demonstrates how machine learning can support strategic workforce decisions by enabling quick salary classification based on historical patterns. It is highly relevant for:

- HR analytics
- Compensation benchmarking
- Workforce planning
- Salary fairness analysis
- Decision support for recruitment and employee evaluation

## Project Strengths

- Real-world dataset and realistic business use case
- Clear data preprocessing pipeline
- Model comparison across multiple algorithms
- Production-style deployment through Streamlit
- Batch and single-record inference support
- Portfolio-ready presentation with measurable results

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Feature importance analysis for model explainability
- Deployment to a cloud backend with API support
- Extend to regression or multi-class income bands
- Add monitoring and retraining pipeline for production use

## Summary

This project is a well-rounded end-to-end machine learning application that covers data preprocessing, feature engineering, model comparison, evaluation, and deployment. With a final XGBoost model accuracy of approximately 87.3% and a clean Streamlit interface, it is an excellent project for technical portfolios, job interviews, and recruiter-facing presentations.

## License

This project is intended for educational and portfolio use.
