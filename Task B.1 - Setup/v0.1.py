# COS30018 - Intelligent Systems
# Task B.1 - Setup
# Name: Faten Jando
# Student ID: 104564296

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import yfinance as yf
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
PREDICTION_DAYS = 60
# ----------------------------------------------LOADING DATA------------------------------------------------------------
data_directory = os.path.expanduser('~/Desktop/COS30018 - Intelligent Systems/Assignment_B/Task B.1 - Set Up/V0.1/stock_data')
data_filename = f"{data_directory}/{COMPANY}_{TRAIN_START}_{TRAIN_END}.csv"

# Check if the data file exists
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

if os.path.exists(data_filename):
    # Load data from CSV if it exists
    data = pd.read_csv(data_filename, index_col='Date', parse_dates=True)
else:
    # Download data if not found locally & save it
    data = yf.download(COMPANY, TRAIN_START, TRAIN_END)
    data.to_csv(data_filename)
# ----------------------------------------------PREPARING DATA----------------------------------------------------------
# Use mid-point of open and close prices
PRICE_VALUE = 'Mid'
data[PRICE_VALUE] = (data['Open'] + data['Close']) / 2

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))

x_train, y_train = [], []
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ------------------------------------------BUILDING THE MODEL----------------------------------------------------------
model_filename = f"{data_directory}/{COMPANY}_model.h5"

# Check if the model already exists
if os.path.exists(model_filename):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_filename)
else:
    # Building the LSTM model layers
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Saving the model after training
    model.save(model_filename)
# ------------------------------------------TESTING THE MODEL-----------------------------------------------------------
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
test_data = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data['Close'].values

# Preparing the test dataset
total_dataset = pd.concat((data[PRICE_VALUE], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line
# EXPLANATION: basically for this part, we take the last PREDICTION_DAYS days of the training data plus all the test
# data to form the model inputs. These inputs are then reshaped to a 2D array (required for scaling) and scaled using
# the same scaler that was used on the training data.

model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines
# EXPLANATION: the test data is prepared in the same way as the training data, forming sequences of PREDICTION_DAYS
# length to predict the next day's price. The data is reshaped to 3D, as required by the LSTM model.

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ------------------------------------------PLOTTING THE RESULTS--------------------------------------------------------
# Plotting actual vs predicted prices
plt.figure(figsize=(14,7))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Candlestick chart
mpf.plot(test_data, type='candle', style='charles', title=f'{COMPANY} Candlestick Chart',
         ylabel='Price', volume=True)

# High/Low chart
plt.figure(figsize=(14,7))
plt.plot(test_data['High'], color="blue", label="High Price")
plt.plot(test_data['Low'], color="red", label="Low Price")
plt.title(f"{COMPANY} High and Low Prices")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()
# -------------------------------------------PREDICT NEXT DAYS-----------------------------------------------------------
# Predicting the next 5 days
num_days_to_predict = 5
future_predictions = []

for _ in range(num_days_to_predict):
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    future_predictions.append(prediction[0,0])
    # Update model_inputs to include the new prediction for future predictions
    model_inputs = np.append(model_inputs, prediction, axis=0)
    model_inputs = model_inputs[1:]

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Predict the next day's price and display it
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction for next day: {prediction}")

# Plotting future predictions
plt.figure(figsize=(14,7))
plt.plot(range(len(actual_prices)), actual_prices, color="black", label="Actual Prices")
plt.plot(range(len(actual_prices), len(actual_prices) + num_days_to_predict), future_predictions, color="blue", label="Predicted Next Days")
plt.title(f"{COMPANY} Share Price Prediction for Next {num_days_to_predict} Days")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()
# -------------------------------------------CONCLUDING REMARKS---------------------------------------------------------
# 1. The predictor is quite bad, especially if you look at the next day prediction, it missed the actual price by about
# 10%-13% can you find the reason?
# --> The prediction errors likely stem from market volatility, limited feature selection (only using the mid-price),
#     a simple model architecture, and potential changes in market conditions that the model wasn’t trained on

# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction gives a much better
# prediction. Even though on the surface, it didn't seem to be a big difference (both use Stacked LSTM). Again, can you
# explain it?
# --> The better performance is due to improved data preprocessing, a more optimised model architecture, and
#     better training techniques.

# 3. A more advanced and quite different technique use CNN to analyse the images of the stock price changes to detect
# some patterns with the trend of the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
# --> Yes, combining LSTM with CNN for pattern recognition, using ensemble methods, adding technical indicators, and
#     exploring advanced architectures like Transformers could significantly enhance prediction accuracy.
