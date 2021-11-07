import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import numpy as np
import pandas as pd
import os

# shuffle two arrays in the same way
def shuffleuni(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

#load data for analysis
def load_data(ticker, n_steps=50, scale=True, lookup_step=1, test_size=0.2, feature_columns=['open', 'close', 'compound']):
    #read file and create storage device
    readfile = os.path.join("csvs", "{}_final.csv".format(ticker))
    df = pd.read_csv(readfile)
    result = {}
    result['df'] = df.copy()

    # mcolumn check to make sure it's legit
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # make date a real thing
    if "Date" not in df.columns:
        df["Date"] = df.index

    #scales all values between 0 and 1 for fast speed
    column_scaler = {}
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler

    # add the MinMaxScaler to the result
    result["column_scaler"] = column_scaler

    # find the future with this magic trick
    df['future'] = df['close'].shift(-lookup_step)

    # get last lookup step before dropping NaN
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)

    #have the sequence storage
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    #add the column with the future predictions
    for entry, target in zip(df[feature_columns + ["Date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get last sequence to figure out future items outside of the range
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence
    
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    #split into data sets by date
    train_samples = int((1 - test_size) * len(X))
    result["X_train"] = X[:train_samples]
    result["y_train"] = y[:train_samples]
    result["X_test"]  = X[train_samples:]
    result["y_test"]  = y[train_samples:]
    # shuffle the datasets for training (if shuffle parameter is set)
    shuffleuni(result["X_train"], result["y_train"])
    shuffleuni(result["X_test"], result["y_test"])

     # get the list of test dates
    dates = result["X_test"][:, -1, -1]
    # get test features from the test dates
    result["test_df"] = result["df"].loc[dates]
    # remove the dupes
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates and convert
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    return result

#create model
def create_model(sequence_length, n_features, units=90, cell=LSTM, n_layers=2, dropout=0.5,
                loss="mean_absolute_error", optimizer="adam"):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            model.add(cell(units, return_sequences=False))
        else:
            model.add(cell(units, return_sequences=True))
        #dropout after layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model