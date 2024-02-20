import warnings

import configs.config as config
from pipeline import pipeline
from utils import get_data, get_train_test_by_exp, transpose_data
import pandas as pd
warnings.filterwarnings("ignore")


def main():
    data = get_data()
    # data = pd.read_csv("data/sales.csv")
    # data = data.astype(int)
    print(data)
    df = transpose_data(data)

    train, test = get_train_test_by_exp(df)

    for cur_filename, cur_model in config.MODELS_TO_RUN:
        print("#" * 10, cur_filename, "#" * 10)
        pipeline(train, test, cur_model, cur_filename)
# /////

if __name__ == "__main__":
    main()
