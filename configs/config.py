import models

# DATA_PATH = "data/Продажи_4 года(2).csv"
DATA_PATH = "data/Продажи.csv"
PREDS_FOLDER = "predictions"
MODEL_PARAMS_PATH = "configs/models_params.json"
TEST_SIZE_MONTHS = 12

# EXPERIMENT_TYPE = "TEST"
EXPERIMENT_TYPE = "PREDICT"

# Для MeanMonthsModel
LAST_N_MONTH = 12

# Для регресионных моделей
SHIFT = 12
USE_WINDOW = True
WINDOW_SIZE = 5

# Список моделей для запуска
# в формате [имя_файла_для_сохранения, модель]
MODELS_TO_RUN = [
    ["MeanForYearsModel", models.MeanForYearsModel],
    ["MeanMonthsModel", models.MeanMonthsModel],
    ["RegressionLinear", models.RegressionLinear],
    ["RegressionRandomForest", models.RegressionRandomForest],
    ["RegressionCatboost", models.RegressionCatboost],
    ["RegressionLogistic", models.RegressionLogistic],
]
