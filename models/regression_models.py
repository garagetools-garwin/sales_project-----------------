import json
from abc import ABC

import numpy as np
import pandas as pd
from tqdm import tqdm

import configs.config as config
from models.feature_generator import FeatureGenerator


class DefaultRegressionModel(ABC):
    def __init__(self) -> None:
        print(f"INIT {self.__class__.__name__}")
        self.feature_generator = None
        self.items = None
        self.test_dates = None
        self.model = None
        self.model_params = None

    def read_model_params(self):
        with open(config.MODEL_PARAMS_PATH) as json_file:
            params = json.load(json_file)
            self.model_params = params[self.__class__.__name__]

    def predict_for_item(self, model, train_y):
        predictions = []
        previous_y = list(train_y)

        for date in self.test_dates:
            row = self.feature_generator.generate_test_features(date, previous_y)
            curr_test = pd.DataFrame([row])

            curr_prediction = int(round(model.predict(curr_test)[0]))
            curr_prediction = max(curr_prediction, 0)
            curr_prediction = min(curr_prediction, 1000000)
            previous_y.append(curr_prediction)
            predictions.append(curr_prediction)

        return list(predictions)

    def fit_predict_for_item(self, item: int) -> list[int]:
        item_df, item_y = self.feature_generator.get_train_data(item)

        try:
            item_model = self.model(**self.model_params)
            item_model.fit(item_df, item_y)

            preds = self.predict_for_item(item_model, item_y)
        except Exception as e:
            print(f"Ошибка с товаром: {item}", e)
            preds = list([0] * config.TEST_SIZE_MONTHS)

        return preds

    def fit(self, train_data: pd.DataFrame) -> None:
        print(f"\tfit {self.__class__.__name__}...")
        self.read_model_params()
        self.feature_generator = FeatureGenerator(train_data)
        self.items = list(train_data["item"].unique())

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        print(f"\tpredict {self.__class__.__name__}...")

        self.test_dates = test_data.index.unique().sort_values()
        test_dates_str = [date.strftime("%Y-%m-%d") for date in self.test_dates]
        preds = np.empty((0, 3))

        for item in tqdm(self.items):
            item_preds = self.fit_predict_for_item(item)

            item_res = np.column_stack(
                (test_dates_str, np.repeat(item, config.TEST_SIZE_MONTHS), item_preds)
            )

            preds = np.vstack((preds, item_res))

        preds = pd.DataFrame(preds, columns=["timestamp", "item", "preds"])
        preds["timestamp"] = pd.to_datetime(preds["timestamp"])
        preds = preds.set_index("timestamp")
        preds = preds.astype({"item": int, "preds": np.int64})
        return preds


class RegressionLinear(DefaultRegressionModel):
    def __init__(self) -> None:
        super().__init__()
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression


class RegressionRandomForest(DefaultRegressionModel):
    def __init__(self) -> None:
        super().__init__()
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor


class RegressionCatboost(DefaultRegressionModel):
    def __init__(self) -> None:
        super().__init__()
        from catboost import CatBoostRegressor

        self.model = CatBoostRegressor


class RegressionLogistic(DefaultRegressionModel):
    def init(self) -> None:
        super().init()
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression
