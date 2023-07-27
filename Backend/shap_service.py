import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
shap.initjs()
import shap

def create_explainer():
    xgb = joblib.load('Model/xgb_model.joblib')
    x_test = joblib.load('Model/x_test_proc_shap.joblib')
    explainer = shap.TreeExplainer(xgb)

    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(x_test)

    shap_df = pd.DataFrame(shap_values, columns = x_test.columns)
    s1 = shap_df.iloc[0]
    class_1 = list((s1[s1>0]).sort_values(ascending=False)[:10].index)
    class_0 = list((s1[s1<0]).sort_values(ascending=True)[:10].index)

    return shap_values, expected_value, x_test, class_0, class_1






