import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "hours-per-week",
    "native-country",
    "capital-gain",
    "capital-loss",
    "educational-num",
    "fnlwgt",
]


def main():
    df = fetch_openml(name="adult", version=2, as_frame=True, parser="auto").frame

    df = df.rename(columns={"sex": "gender", "education-num": "educational-num"})

    for col in df.columns:
        df[col] = df[col].replace("?", np.nan)

    df["income"] = df["class"].map({"<=50K": 0, ">50K": 1}).astype(int)

    X = df[FEATURES]
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

    joblib.dump(model, "xgboost_salary_model.pkl")
    print("Saved model to xgboost_salary_model.pkl")


if __name__ == "__main__":
    main()
