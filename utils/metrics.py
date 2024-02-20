import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

CALCULATED_METRICS = {"RMSE": mean_squared_error}


def common_metrics(df_test: pd.DataFrame, df_preds: pd.DataFrame):
    y_true, y_pred = df_test["sales"], df_preds["preds"]

    return {
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }


def mean_metrics(df_test: pd.DataFrame, df_preds: pd.DataFrame):
    items = list(df_test["item"].unique())

    rmse_list = []
    mae_list = []
    mape_list = []

    for item in items:
        y_true = list(df_test.loc[df_test["item"] == item, "sales"].sort_index())
        y_pred = list(df_preds.loc[df_preds["item"] == item, "preds"].sort_index())

        rmse_list.append(mean_squared_error(y_true, y_pred, squared=False))
        mae_list.append(mean_absolute_error(y_true, y_pred))
        mape_list.append(mean_absolute_percentage_error(y_true, y_pred))

    return {"RMSE": float(np.mean(rmse_list)), 
            "MAE": float(np.mean(mae_list)),
            "MAPE": float(np.mean(mae_list)),}


def calc_metrics(
    df_test: pd.DataFrame, df_preds: pd.DataFrame, print_metrics=False
) -> dict:
    print("Расчет метрик...")

    metrics = {
        "common": common_metrics(df_test, df_preds),
        "mean": mean_metrics(df_test, df_preds),
    }

    if print_metrics:
        pprint.pprint(metrics)

    return metrics
