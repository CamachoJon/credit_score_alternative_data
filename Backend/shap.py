import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
shap.initjs()


def create_explainer():
    xgb = joblib.load('Model/xgboost.joblib')
    x_test = joblib.load('Model/xtest_shap.joblib')
    explainer = shap.TreeExplainer(xgb)

    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(x_test)

    return shap_values, expected_value, x_test






