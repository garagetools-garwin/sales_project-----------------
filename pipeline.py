import pandas as pd

import configs.config as config
from utils import calc_metrics, save_prediction


def pipeline(train: pd.DataFrame, test: pd.DataFrame, model_class, filename: str):
    model = model_class()
    model.fit(train)
    preds = model.predict(test)
    save_prediction(preds, filename)
    if config.EXPERIMENT_TYPE == "TEST":
        metrics = calc_metrics(test, preds, print_metrics=True)
