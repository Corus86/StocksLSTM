import numpy as np
import matplotlib.pyplot as plt
from predict import create_model, load_data
from param import *

#Red is predicted blue is the real data
def plot_graph(test_df):
    plt.plot(test_df[f'realclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'predictclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

#create the final data frame with the predicted values inserted
def get_final_df(model, data):
    X_test = data["X_test"]
    y_test = data["y_test"]
    # predict and give us the monies
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add the predictions
    test_df[f"predictclose_{LOOKUP_STEP}"] = y_pred
    # add the real ones
    test_df[f"realclose_{LOOKUP_STEP}"] = y_test
    # sort the dates
    test_df.sort_index(inplace=True)
    final_df = test_df

    return final_df


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    if SCALE:
        predicted_price = data["column_scaler"]["close"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# load the data
data = load_data(ticker, N_STEPS, scale=SCALE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER)

# load optimal model weights from results folder
model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)

# evaluate the model
loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
if SCALE:
    mean_absolute_error = data["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae

# get the final dataframe for the testing set
final_df = get_final_df(model, data)
# predict the future price
future_price = predict(model, data)
# printing metrics
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(15))
# save the final dataframe to csv-results folder
csv_filename = os.path.join(model_name + "MEME.csv")
final_df.to_csv(csv_filename)