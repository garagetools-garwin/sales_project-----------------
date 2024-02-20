import numpy as np
import pandas as pd

import configs.config as config


def train_test_split_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(df.columns)

    train = df[cols[: -config.TEST_SIZE_MONTHS]]
    test = df[[cols[0]] + cols[-config.TEST_SIZE_MONTHS :]]

    return train, test


def train_test_split_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = df.index.unique().sort_values()

    train_dates = all_dates[: -config.TEST_SIZE_MONTHS]
    test_dates = all_dates[-config.TEST_SIZE_MONTHS :]

    train = df[df.index.isin(train_dates)]
    test = df[df.index.isin(test_dates)]

    return train, test


def get_train_test_by_exp(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Формирование обучающего и тестового датасета...")
    if config.EXPERIMENT_TYPE == "TEST":
        return train_test_split_long(df)

    # config.EXPERIMENT_TYPE == "PREDICT":
    train = df.copy()

    max_date = train.index.max()
    test_dates = [
        max_date + pd.DateOffset(months=i)
        for i in range(1, config.TEST_SIZE_MONTHS + 1)
    ]

    test = np.empty((0, 2))
    items = list(train["item"].unique())

    for item in items:
        item_test = np.column_stack(
            (test_dates, np.repeat(item, config.TEST_SIZE_MONTHS))
        )
        test = np.vstack((test, item_test))

    test = pd.DataFrame(test, columns=["timestamp", "item"]).set_index("timestamp")

    return train, test
