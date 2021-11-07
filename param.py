import os
from tensorflow.keras.layers import LSTM

N_STEPS = 50
LOOKUP_STEP = 15

SCALE = True
SHUFFLE = True
SPLIT_BY_DATE = False
TEST_SIZE = 0.3
FEATURE_COLUMNS = ["open", "close", "compound"]

N_LAYERS = 2
CELL = LSTM
UNITS = 90
DROPOUT = 0.5
BIDIRECTIONAL = False

LOSS = "mean_absolute_error"
OPTIMIZER = "adam"
BATCH_SIZE = 50
EPOCHS = 100

ticker = "Apple"
ticker_data_filename = os.path.join("data", f"{ticker}.csv")
model_name = f"Apple_final"