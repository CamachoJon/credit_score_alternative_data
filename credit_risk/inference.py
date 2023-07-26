import os
import pandas as pd
import numpy as np
import joblib
from credit_risk import FEATURES
from credit_risk import MODEL_DIR
from credit_risk import preprocess


def make_predictions(test_df: pd.DataFrame) -> np.ndarray:
    """
    Makes prediction for the inference data and returns the credit risk classification
    :param test_df:
    :return: array[Defaulter(1) / Non-Defaulter(0)]
    """
    X = test_df[FEATURES]

    X_processed = preprocess.feature_engineering(X, training=False)

    xg_model_path = os.path.join(MODEL_DIR, "xgboost.joblib")
    xg_model = joblib.load(xg_model_path)

    y_pred_inference = xg_model.predict(X_processed)

    return y_pred_inference