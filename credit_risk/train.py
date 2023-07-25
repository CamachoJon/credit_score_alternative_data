import os
import pandas as pd
import numpy as np
import joblib
from credit_risk import FEATURES
from credit_risk import TARGET
from credit_risk import MODEL_DIR
from credit_risk.preprocess import feature_engineering
from typing import Dict

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray,
                 precision: int = 2):
    """
    Compute the performance metrics for the test and predicted samples
    :param y_test:
    :param y_pred:
    :param precision: 2
    :return: Accuracy, roc_auc score and confusion matrix
    """

    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return round(accuracy, precision), round(roc_auc, precision), cm


def build_model(df: pd.DataFrame) -> Dict[str, float]:
    """
    Train the model and returns a dictionary with the model performances
    (for example {'accuracy': 0.8})
    :param df:
    :return: {"acc": float, "roc_auc": float, "confusion_matrix": np.ndarray}
    """
    df = df[df['CODE_GENDER'] != 'XNA']
    class_zero = df[df['TARGET'] == 0].sample(24825,random_state=42)
    class_one = df[df['TARGET'] == 1].sample(24825, random_state=42)

    df = pd.concat([class_zero, class_one])

    X = df[FEATURES]
    y = df[[TARGET]]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    processed_X_train = feature_engineering(X_train)
    processed_X_test = feature_engineering(X_test, training=False)

    print("Train shape: ",processed_X_train.shape)
    print("Test shape: ", processed_X_test.shape, "\n")

    # Create an instance of the XGBoost model
    xg_model = xgb.XGBClassifier(#use_label_encoder=False,
                            objective='binary:logistic',
                            eval_metric='logloss',
                            n_estimators=100,
                            max_depth=5,
                            min_child_weight=1,
                            gamma=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            learning_rate=0.1)

    # Fit the model to the training data
    xg_model.fit(processed_X_train, y_train)

    joblib.dump(xg_model, filename=os.path.join(MODEL_DIR, "xgboost.joblib"))

    y_pred = xg_model.predict(processed_X_test) 

    acc, roc, cm = compute_metrics(y_test, y_pred)

    return {"accuracy": acc, "roc_auc":roc, "confusion_matrix":cm}