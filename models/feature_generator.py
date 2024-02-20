import numpy as np
import pandas as pd

import configs.config as config


class FeatureGenerator:
    def __init__(self, train_data) -> None:
        self.shifts = config.SHIFT
        self.use_window = config.USE_WINDOW
        self.window_size = config.WINDOW_SIZE

        self.train_data = train_data

    def get_train_data(self, item):
        y_series = self.train_data.loc[self.train_data["item"] == item, "sales"]
        return self.generate_train_features(y_series)

    def create_date_features(self, date):
        """Создает фичи из даты"""

        row = {}
        # row["dayofweek"] = date.dayofweek
        # row["quarter"] = date.quarter
        row["month"] = date.month
        row["year"] = date.year
        # row["dayofyear"] = date.dayofyear
        # row["dayofmonth"] = date.day
        # row["weekofyear"] = date.weekofyear
        return row

    def generate_train_features(self, y_series):

        ##### DATES #####
        df = pd.DataFrame([self.create_date_features(date) for date in y_series.index])

        ##### SHIFTS #####
        df.index = y_series.index

        for shift in range(1, self.shifts + 1):
            df[f"shift_{shift}"] = y_series.shift(shift, axis=0)

        y = y_series.copy()

        drop_indices = df.index[df.isna().sum(axis=1) > 0]
        df = df.drop(index=drop_indices)
        y = y.drop(index=drop_indices)

        ##### ROLLING WINDOWS #####
        if self.use_window:
            df["rolling_mean"] = (
                y.rolling(self.window_size, min_periods=1).mean().shift(1, axis=0)
            )
            df["rolling_max"] = (
                y.rolling(self.window_size, min_periods=1).max().shift(1, axis=0)
            )
            df["rolling_min"] = (
                y.rolling(self.window_size, min_periods=1).min().shift(1, axis=0)
            )

            drop_indices = df.index[df.isna().sum(axis=1) > 0]
            df = df.drop(index=drop_indices)
            y = y.drop(index=drop_indices)

        df = df.reset_index(drop=True)

        return df, y

    def generate_test_features(self, date, previous_y):
        row = self.create_date_features(date)

        for shift in range(1, self.shifts + 1):
            row[f"shift_{shift}"] = previous_y[-1 * shift]

        if self.use_window:
            row["rolling_mean"] = np.mean(previous_y[-self.window_size :])
            row["rolling_max"] = np.max(previous_y[-self.window_size :])
            row["rolling_min"] = np.min(previous_y[-self.window_size :])

        return row
