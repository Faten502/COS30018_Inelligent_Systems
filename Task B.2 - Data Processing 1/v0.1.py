# COS30018 - Intelligent Systems
# Task B.2 - Data Processing 1
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
from sklearn.model_selection import train_test_split

# ------------------------------------------FUNCTION FOR LOADING & PROCESSING-------------------------------------------
def load_and_process_data(ticker, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Volume'],
                          handle_nan=True, nan_strategy='drop', split_ratio=0.8, split_method='date',
                          random_state=42, save_local=True, load_local=False, scale_features=True):
    """
    The load_and_process_data function will load and process the stock data. The function can also handle missing values,
    it can split the data into training and testing sets, scale the features, and allow for saving/loading the data
    locally for future use.

    Parameters:
    - ticker: str, the stock ticker symbol (e.g. 'CBA.AX').
    - start_date: str, start date for the data (format: 'YYYY-MM-DD').
    - end_date: str, end date for the data (format: 'YYYY-MM-DD').
    - features: list, list of features (columns) to include (e.g., ['Open', 'Close']).
    - handle_nan: bool, whether to handle NaN values (default is True).
    - nan_strategy: str, how to handle NaN ('drop' to remove NaNs, 'fill' to forward-fill them).
    - split_ratio: float, ratio for splitting data into training/testing sets (default is 0.8).
    - split_method: str, how to split the data ('date' for time-based split, 'random' for random split).
    - random_state: int, seed for reproducibility in random operations (default is 42).
    - save_local: bool, whether to save the processed data locally (default is True).
    - load_local: bool, whether to load data from local storage if available (default is False).
    - scale_features: bool, whether to scale feature columns using MinMaxScaler (default is True).

    Returns:
    - X_train, X_test, y_train, y_test: numpy arrays of processed and split data for training/testing.
    - scalers: dictionary containing the scalers used for each feature.
    - data_directory: str, path to the directory where data is saved or loaded from.
    """
    # Setting the directory and filename for saving/loading the dataset locally.
    data_directory = os.path.expanduser('~/Desktop/COS30018 - Intelligent Systems/Assignment_B/Task B.2 - Data processing 1')
    data_filename = f"{data_directory}/{ticker}_{start_date}_{end_date}.csv"

    # Load data from local storage if it exists and 'load_local' is set to True.
    if load_local and os.path.exists(data_filename):
        data = pd.read_csv(data_filename, index_col='Date', parse_dates=True)
    else:
        # Download data from Yahoo Finance using yfinance if not loading locally.
        data = yf.download(ticker, start=start_date, end=end_date)
        if save_local:
            # Save the downloaded data locally if 'save_local' is True.
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
            data.to_csv(data_filename)

    # Handle missing data according to the chosen strategy (drop or fill NaNs).
    if handle_nan:
        if nan_strategy == 'drop':
            data.dropna(inplace=True)
        elif nan_strategy == 'fill':
            data.fillna(method='ffill', inplace=True)

    # Select only the specified features from the dataset.
    data = data[features]

    # Convert all data to float type to avoid potential dtype issues.
    data = data.astype(float)

    # Initialise a dictionary to store scalers for each feature, if scaling is enabled.
    scalers = {}
    if scale_features:
        for feature in features:
            scaler = MinMaxScaler()
            data.loc[:, feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            scalers[feature] = scaler

    # Split the data into training and testing sets based on the chosen split method.
    if split_method == 'date':
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
    else:
        train_data, test_data = train_test_split(data, test_size=1 - split_ratio, random_state=random_state, shuffle=True)

    # Prepare the training and testing datasets by creating sequences of 'PREDICTION_DAYS' length.
    X_train, y_train = [], []
    X_test, y_test = [], []

    PREDICTION_DAYS = 60

    # Create the training sequences (X_train) and corresponding target values (y_train).
    for x in range(PREDICTION_DAYS, len(train_data)):
        X_train.append(train_data.iloc[x - PREDICTION_DAYS:x].values)
        y_train.append(train_data.iloc[x]['Close'])

    # Create the testing sequences (X_test) and corresponding target values (y_test).
    for x in range(PREDICTION_DAYS, len(test_data)):
        X_test.append(test_data.iloc[x - PREDICTION_DAYS:x].values)
        y_test.append(test_data.iloc[x]['Close'])

    # Convert the sequences to numpy arrays to facilitate use with TensorFlow/Keras.
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), scalers, data_directory

# -----------------------------------------------------MAIN SCRIPT------------------------------------------------------
# My constants and parameters for loading, processing, and modeling the data.
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
PREDICTION_DAYS = 60

# My 'load_and_process_data' function to load, process, and split the data.
X_train, y_train, X_test, y_test, scalers, data_directory = load_and_process_data(COMPANY, TRAIN_START, TRAIN_END,
                                                                                  features=['Open', 'High', 'Low', 'Close', 'Volume'],
                                                                                  handle_nan=True, nan_strategy='drop',
                                                                                  split_ratio=0.8, split_method='date',
                                                                                  save_local=True, load_local=False,
                                                                                  scale_features=True)

# Reshape the data to fit the input shape expected by the LSTM model.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(['Open', 'High', 'Low', 'Close', 'Volume'])))

# ------------------------------------------BUILDING THE MODEL----------------------------------------------------------
# Define the model's filename for saving/loading the trained model.
model_filename = f"{data_directory}/{COMPANY}_model.h5"

# Check if the model already exists, if yes load it --> if not, build and train a new model.
if os.path.exists(model_filename):
    model = tf.keras.models.load_model(model_filename)
else:
    # Build the LSTM model with multiple layers.
    model = Sequential()

    # Add LSTM layers with dropout for regularisation to prevent overfitting.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Add a Dense layer with one unit (output prediction of the next 'Close' price).
    model.add(Dense(units=1))

    # Compile the model with Adam optimizer and mean squared error as the loss function.
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model on the training data.
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Save the trained model for future use.
    model.save(model_filename)

# ------------------------------------------TESTING THE MODEL-----------------------------------------------------------
# Download the test dataset to evaluate the model's performance.
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
test_data = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data['Close'].values

# Prepare the combined dataset for testing, including both training and new test data.
total_dataset = pd.concat((pd.DataFrame(y_train, columns=['Close']), test_data[['Open', 'High', 'Low', 'Close', 'Volume']]), axis=0)

# Apply the stored scalers to the features in the combined dataset.
for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
    total_dataset[feature] = scalers[feature].transform(total_dataset[feature].values.reshape(-1, 1))

# Extract the model input sequences from the total dataset.
model_inputs = total_dataset[len(total_dataset) - len(test_data) - 60:].values

# Prepare the test sequences for prediction.
x_test = []
for x in range(60, len(model_inputs)):
    x_test.append(model_inputs[x - 60:x])

x_test = np.array(x_test)  # Convert the list of sequences into a numpy array.

# Predict the prices using the trained model.
predicted_prices = model.predict(x_test)
predicted_prices = scalers['Close'].inverse_transform(predicted_prices)  # Inverse transform to get the original price scale.

# ------------------------------------------PLOTTING THE RESULTS--------------------------------------------------------
# Plot the actual vs predicted prices.
plt.figure(figsize=(14,7))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Candlestick chart for the test period.
mpf.plot(test_data, type='candle', style='charles', title=f'{COMPANY} Candlestick Chart',
         ylabel='Price', volume=True)

# Plotting high and low prices for the test period.
plt.figure(figsize=(14,7))
plt.plot(test_data['High'], color="blue", label="High Price")
plt.plot(test_data['Low'], color="red", label="Low Price")
plt.title(f"{COMPANY} High and Low Prices")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()

# -------------------------------------------PREDICT NEXT DAYS----------------------------------------------------------
# Predicting the next 5 days.
num_days_to_predict = 5
future_predictions = []

for _ in range(num_days_to_predict):
    # Prepare the latest data sequence for prediction.
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:]]
    real_data = np.array(real_data)

    prediction = model.predict(real_data)
    future_predictions.append(prediction[0, 0])

    # Update model_inputs to include the new prediction for the next iteration.
    new_data = np.zeros((1, 60, 5))
    new_data[:, :, :-1] = model_inputs[-1, 1:]
    new_data[:, -1, -1] = prediction[0, 0]

    model_inputs = np.append(model_inputs, new_data, axis=0)
    model_inputs = model_inputs[1:]  # Remove the oldest entry to keep the sequence length consistent.

# Inverse transform the predictions to get them back to the original price scale.
future_predictions = scalers['Close'].inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Predict the next day's price and display it.
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scalers['Close'].inverse_transform(prediction)
print(f"Prediction for next day: {prediction}")

# Plot the predicted future prices for the next 5 days.
plt.figure(figsize=(14,7))
plt.plot(range(len(actual_prices)), actual_prices, color="black", label="Actual Prices")
plt.plot(range(len(actual_prices), len(actual_prices) + num_days_to_predict), future_predictions, color="blue", label="Predicted Next Days")
plt.title(f"{COMPANY} Share Price Prediction for Next {num_days_to_predict} Days")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()
