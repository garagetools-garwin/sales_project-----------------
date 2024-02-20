import pandas as pd

import configs.config as config


def preds_to_orig_view(df: pd.DataFrame) -> pd.DataFrame:
    data_list = []
    items = list(df["item"].unique())
    dates = list(df.index.unique().sort_values())
    dates = [date.strftime("%Y-%m-%d") for date in dates]

    for item in items:
        vals = list(df.loc[df["item"] == item].sort_values(by="timestamp")["preds"])
        data_list.append([item] + vals)

    return pd.DataFrame(data_list, columns=["item"] + dates)


def save_prediction(data: pd.DataFrame, filename: str) -> None:
    print("Сохранение предсказаний...")
    df = preds_to_orig_view(data)
    df.to_csv(f"{config.PREDS_FOLDER}/{filename}.csv", index=False)
