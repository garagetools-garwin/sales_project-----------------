import pandas as pd

import configs.config as config


class MeanForYearsModel:
    def __init__(self) -> None:
        print(f"INIT {self.__class__.__name__}")
        self.mean_sales = None

    def fit(self, train_data: pd.DataFrame) -> None:
        print(f"\tfit {self.__class__.__name__}...")
        temp_data = train_data.copy()
        temp_data["month"] = train_data.index.month

        # self.mean_sales = temp_data.groupby(["item", "month"])["sales"].mean().round()
        self.mean_sales = temp_data.groupby(["item", "month"])["sales"].mean()
        

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        print(f"\tpredict {self.__class__.__name__}...")

        df_preds = test_data.copy()
        preds_list = []

        for index, row in test_data.iterrows():
            # pred = int(self.mean_sales.loc[(row["item"], index.month)])
            pred = int(round(self.mean_sales.loc[(row["item"], index.month)]))
            preds_list.append(pred)

        df_preds["preds"] = preds_list

        return df_preds


class MeanMonthsModel:
    def __init__(self) -> None:
        print(f"INIT {self.__class__.__name__}")
        self.LAST_N_MONTH = config.LAST_N_MONTH
        self.mean_by_item = None

    def fit(self, train_data: pd.DataFrame) -> None:
        print(f"\tfit {self.__class__.__name__}...")
        self.mean_by_item = {}
        items = list(train_data["item"].unique())
        train_dates = list(train_data.index.unique().sort_values())

        for item in items:
            n_month_mean = train_data.loc[
                (train_data["item"] == item)
                & (train_data.index >= train_dates[-self.LAST_N_MONTH]),
                "sales",
            ].mean()
            self.mean_by_item[item] = int(round(n_month_mean))

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        print(f"\tpredict {self.__class__.__name__}...")

        df_preds = test_data.copy()
        preds_list = []

        for _, row in test_data.iterrows():
            preds_list.append(self.mean_by_item[row["item"]])

        df_preds["preds"] = preds_list

        return df_preds
