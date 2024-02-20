import locale
from datetime import datetime

import pandas as pd

import configs.config as config


def get_data() -> pd.DataFrame:
    print("Чтение данных...")
    df = pd.read_csv(config.DATA_PATH, delimiter=";")
    # df = df.iloc[:10]  ### НЕ ЗАБЫТЬ УБРАТЬ
    # df = df[
    #     [
    #         'Код "Инфор"',
    #         "май.19",
    #         "июн.19",
    #         "июл.19",
    #         "авг.19",
    #         "сен.19",
    #         "окт.19",
    #         "ноя.19",
    #         "дек.19",
    #         "янв.20",
    #         "фев.20",
    #         "мар.20",
    #         "апр.20",
    #         "май.20",
    #         "июн.20",
    #         "июл.20",
    #         "авг.20",
    #         "сен.20",
    #         "окт.20",
    #         "ноя.20",
    #         "дек.20",
    #         "янв.21",
    #         "фев.21",
    #         "мар.21",
    #         "апр.21",
    #         "май.21",
    #         "июн.21",
    #         "июл.21",
    #         "авг.21",
    #         "сен.21",
    #         "окт.21",
    #         "ноя.21",
    #         "дек.21",
    #         "янв.22",
    #         "фев.22",
    #         "мар.22",
    #         "апр.22",
    #         "май.22",
    #         "июн.22",
    #         "июл.22",
    #         "авг.22",
    #         "сен.22",
    #         "окт.22",
    #         "ноя.22",
    #         "дек.22",
    #         "янв.23",
    #         "фев.23",
    #         "мар.23",
    #         "апр.23",
    #         "май.23",
    #     ]
    # ]
    df = df[[
        'Номенклатура.Код_Инфор', 'Февраль 2020', 'Март 2020', 'Апрель 2020', 'Май 2020',
       'Июнь 2020', 'Июль 2020', 'Август 2020', 'Сентябрь 2020',
       'Октябрь 2020', 'Ноябрь 2020', 'Декабрь 2020', 'Январь 2021',
       'Февраль 2021', 'Март 2021', 'Апрель 2021', 'Май 2021', 'Июнь 2021',
       'Июль 2021', 'Август 2021', 'Сентябрь 2021', 'Октябрь 2021',
       'Ноябрь 2021', 'Декабрь 2021', 'Январь 2022', 'Февраль 2022',
       'Март 2022', 'Апрель 2022', 'Май 2022', 'Июнь 2022', 'Июль 2022',
       'Август 2022', 'Сентябрь 2022','Октябрь 2022', 'Ноябрь 2022', 'Декабрь 2022', 'Январь 2023',
       'Февраль 2023', 'Март 2023', 'Апрель 2023', 'Май 2023', 'Июнь 2023', 'Июль 2023', 'Август 2023','Сентябрь 2023',
       'Октябрь 2023', 'Ноябрь 2023', 'Декабрь 2023', 'Январь 2024'
    ]]
    df = df.fillna(0)

    # locale.setlocale(locale.LC_TIME, "ru_RU.UTF-8")

    # def convert_date(dt):
    #     dt = dt.replace("май", "мая")
    #     dt = dt.replace("Октябрь", "Октября")
    #     # return datetime.strptime(dt, "%b.%y").strftime("%Y-%m-%d")

    def convert_date(dt):
        match = {
            "Январь": "january",
            "Февраль": "february",
            "Март": "march",
            "Апрель": "april",
            "Май": "may",
            "Июнь": "june",
            "Июль": "july",
            "Август": "august",
            "Сентябрь": "september",
            "Октябрь": "october",
            "Ноябрь": "november",
            "Декабрь": "december"
        }
        for key, val in match.items():
            dt = dt.replace(key, val)
        
        # return datetime.strptime(dt, "%b.%y").strftime("%Y-%m-%d")
        return datetime.strptime(dt, "%B %Y").strftime("%Y-%m-%d")

    df.columns = ["item"] + [convert_date(col) for col in df.columns[1:]]

    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: str(x).replace(",", "."))
        df[col] = df[col].apply(lambda x: str(x).replace(" ", ""))
        df[col] = df[col].astype(float)
        df[col] = df[col].astype(int)
        df[col] = df[col].apply(lambda x: x if x >= 0 else 0)

    df["item"] = df["item"].astype(int)

    df.to_csv("data/sales.csv", index=False)

    return df


def transpose_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Преобразование датасета в длинный формат...")
    items = list(df["item"])

    df_t = pd.DataFrame(columns=["timestamp", "sales", "item"])

    for item in items:
        item_df = df.loc[df["item"] == item, df.columns[1:]].T.reset_index()
        item_df["item"] = item
        item_df.columns = ["timestamp", "sales", "item"]

        df_t = pd.concat([df_t, item_df], axis=0)

    df_t = df_t[["timestamp", "item", "sales"]]
    df_t["timestamp"] = pd.to_datetime(df_t["timestamp"])
    df_t = df_t.set_index("timestamp")

    return df_t
