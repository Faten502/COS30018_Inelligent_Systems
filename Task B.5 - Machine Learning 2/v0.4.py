# COS30018 - Intelligent Systems
# Task B.5 - Machine Learning 2
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
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanAbsoluteError

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
    data_directory = os.path.expanduser(
        '~/Desktop/COS30018 - Intelligent Systems/Assignment_B/Task B.5 - Machine Learning 2')
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

    # Convert all data to float type to avoid potential dtype issues.
    data = data[features].astype(float)

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
        train_data, test_data = train_test_split(data, test_size=1 - split_ratio, random_state=random_state,
                                                 shuffle=True)

    # Prepare the training and testing datasets by creating sequences of 'PREDICTION_DAYS' length.
    X_train, y_train, X_test, y_test = [], [], [], []
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

# -----------------------------------------------MODEL CREATION FUNCTION------------------------------------------------
def create_dl_model(input_shape, n_layers=3, units=[50, 100, 150], layer_names=['LSTM', 'GRU', 'RNN'],
                    dropout_rate=0.3, loss='mean_absolute_error', optimizer='adam', bidirectional=False,
                    metrics=[MeanAbsoluteError()]):
    """
    Creates a deep learning model based on the specified architecture and hyperparameters.

    Parameters:
    - input_shape: tuple, shape of the input data (e.g., (sequence_length, n_features)).
    - n_layers: int, number of layers to add to the model.
    - units: list of int, number of units (neurons) per layer.
    - layer_names: list of str, type of layer to use per layer ('LSTM', 'GRU', or 'RNN').
    - dropout_rate: float, dropout rate for regularization (default 0.3).
    - loss: str, loss function to use (e.g., 'mean_absolute_error').
    - optimizer: str, optimizer to use (e.g., 'adam').
    - bidirectional: bool, whether to use bidirectional layers.
    - metrics: list, list of metrics to evaluate during training.

    Returns:
    - model: Keras Sequential model compiled and ready for training.
    """
    assert len(units) == n_layers, "The length of 'units' must equal 'n_layers'."
    assert len(layer_names) == n_layers, "The length of 'layer_names' must equal 'n_layers'."

    model = Sequential()

    for i in range(n_layers):
        layer_type = layer_names[i]
        unit = units[i]
        if layer_type == 'LSTM':
            layer_class = LSTM
        elif layer_type == 'GRU':
            layer_class = GRU
        elif layer_type == 'RNN':
            layer_class = SimpleRNN
        else:
            raise ValueError("Invalid layer type. Choose from 'LSTM', 'GRU', or 'RNN'.")

        return_sequences = True if i < n_layers - 1 else False

        # Only set input_shape for the first layer
        layer_kwargs = {
            'units': unit,
            'return_sequences': return_sequences,
        }
        if i == 0:
            layer_kwargs['input_shape'] = input_shape

        if bidirectional:
            layer = Bidirectional(layer_class(**layer_kwargs))
        else:
            layer = layer_class(**layer_kwargs)

        model.add(layer)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
# ------------------------------------------EXPERIMENTING WITH HYPERPARAMETERS-------------------------------------------
def experiment_with_hyperparameters(X_train, y_train, X_test, y_test, hyperparameter_configurations):
    """
    Function to train models using different configurations of hyperparameters and evaluate their performance.

    Parameters:
    - X_train, y_train: Training data (features and target).
    - X_test, y_test: Testing data (features and target).
    - hyperparameter_configurations: List of dicts, each containing a set of hyperparameters.

    Returns:
    - results: List of dictionaries containing the final validation loss for each configuration.
    - validation_histories: List of validation loss values (per epoch) for each configuration.
    """

    results = []  # List to store final validation loss for each model.
    validation_histories = []  # List to store loss history (per epoch) for each model.

    # Iterate through each hyperparameter configuration.
    for config in hyperparameter_configurations:
        print(f"Training model with config: {config}")  # Log the current configuration being tested.

        # Extract parameters from the configuration dict.
        num_layers = config['num_layers']
        units_per_layer = config['units_per_layer']
        layer_type = config['layer_type']
        dropout_rate = config.get('dropout_rate', 0.2)  # Default dropout rate is 0.2 if not provided.
        epochs = config['epochs']
        batch_size = config['batch_size']
        use_bidirectional = config.get('use_bidirectional', False)
        use_batchnorm = config.get('use_batchnorm', False)
        activation = config.get('activation', None)

        # Create the model using the extracted parameters.
        input_shape = (X_train.shape[1], X_train.shape[2])  # Input shape based on the number of past days and features.
        model = create_dl_model(input_shape, num_layers, units_per_layer, layer_type, dropout_rate, use_bidirectional,
                                use_batchnorm, activation)

        # Train the model using the training data and validate on the test data.
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                            verbose=1)

        # Extract the final validation loss after training.
        final_loss = history.history['val_loss'][-1]
        print(f"Final validation loss: {final_loss}")

        # Save the result (configuration + final validation loss).
        results.append({
            'config': config,
            'val_loss': final_loss
        })

        # Store the validation loss history (for plotting later).
        validation_histories.append(history.history['val_loss'])

    return results, validation_histories

# --------------------------------------MULTISTEP AND MULTIVARIATE PREDICTION FUNCTIONS---------------------------------
def multistep_prediction(model, last_sequence, scalers, days=5):
    """
    Predicts multiple future steps ahead using the last available data sequence.

    Parameters:
    - model: Trained deep learning model.
    - last_sequence: The last sequence from the test data.
    - scalers: Dictionary of scalers used to inverse transform the prediction.
    - days: The number of future days to predict.

    Returns:
    - future_predictions: List of predicted closing prices for 'days' future days.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    # Index of the 'Close' feature
    close_feature_index = list(scalers.keys()).index('Close')

    for _ in range(days):
        # Reshape to match model input shape
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        # Extract scalar value from prediction
        prediction_value = prediction[0, 0]

        # Inverse transform to get the actual price
        predicted_price = scalers['Close'].inverse_transform([[prediction_value]])
        future_predictions.append(predicted_price[0, 0])

        # Prepare next input sequence
        next_step = current_sequence[-1, :].copy()
        next_step[close_feature_index] = prediction_value  # Update 'Close' with predicted value

        # Append new step and remove the oldest
        current_sequence = np.vstack((current_sequence[1:], next_step))

    return future_predictions


def multivariate_prediction(model, X_test, scalers):
    """
    Predicts the closing price using multiple features (multivariate prediction).

    Parameters:
    - model: Trained deep learning model.
    - X_test: The input data for making a prediction.
    - scalers: Dictionary of scalers used to inverse transform the prediction.

    Returns:
    - predicted_price: The predicted closing price for the given input.
    """
    prediction = model.predict(np.array([X_test[-1]]))
    predicted_price = scalers['Close'].inverse_transform(prediction)
    return predicted_price[0, 0]

def multivariate_multistep_prediction(model, X_test, scalers, days=5):
    """
    This function solves the multivariate, multistep prediction problem, predicting the future closing
    prices for 'days' future days using multiple features.

    Parameters:
    - model: Trained deep learning model.
    - X_test: The latest input data to make predictions (must be the same shape as model input).
    - scalers: Dictionary of scalers used to inverse transform the prediction.
    - days: The number of future days to predict.

    Returns:
    - future_predictions: List of predicted closing prices for 'days' future days.
    """
    future_predictions = []

    for _ in range(days):
        # Predict the next day's closing price
        prediction = model.predict(np.array([X_test[-1]]))
        predicted_price = scalers['Close'].inverse_transform(prediction)
        future_predictions.append(predicted_price[0, 0])

        # Append the predicted price to the input data and slide the window
        next_input = np.append(X_test[-1][1:], prediction, axis=0)
        X_test = np.append(X_test, [next_input], axis=0)

    return future_predictions
# -----------------------------------------VISUALISING RESULTS---------------------------------------------------------
def plot_validation_loss(results, validation_histories):
    """
    Plots the validation loss per epoch for each model configuration and compares the final losses.

    Parameters:
    - results: List of final validation losses for each model.
    - validation_histories: List of validation loss values per epoch for each model.
    """

    # Plot validation loss across epochs for each configuration.
    plt.figure(figsize=(10, 6))
    for idx, history in enumerate(validation_histories):
        plt.plot(history,
                 label=f"Config {idx + 1}: {results[idx]['config']['layer_type']} - Layers: {results[idx]['config']['num_layers']}")
    plt.title("Validation Loss per Epoch for Each Configuration")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.show()

    # Plot a bar chart to compare final validation losses for each configuration.
    configs = [f"Config {i + 1}" for i in range(len(results))]
    final_losses = [result['val_loss'] for result in results]

    plt.figure(figsize=(10, 6))
    plt.bar(configs, final_losses, color='skyblue')
    plt.title("Final Validation Loss for Each Configuration")
    plt.ylabel("Validation Loss")
    plt.xlabel("Configurations")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------------------------------MAIN SCRIPT------------------------------------------------------
# My constants and parameters for loading, processing, and modeling the data.
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-08-01'
PREDICTION_DAYS = 60  # sequence_length

# Load, process, and split the data.
X_train, y_train, X_test, y_test, scalers, data_directory = load_and_process_data(
    COMPANY, TRAIN_START, TRAIN_END,
    features=['Open', 'High', 'Low', 'Close', 'Volume'],
    handle_nan=True, nan_strategy='drop', split_ratio=0.8, split_method='date',
    save_local=True, load_local=False, scale_features=True
)

# Determine the number of features based on selected columns
n_features = len(['Open', 'High', 'Low', 'Close', 'Volume'])

# Reshape the data to fit the input shape expected by the model.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

# Set up the model parameters
k_days = PREDICTION_DAYS  # sequence_length
units = [50, 100, 150]    # Adjust units per layer as needed
layer_names = ['LSTM', 'GRU', 'RNN']
n_layers = len(units)
dropout_rate = 0.3
loss = 'mean_absolute_error'
optimizer = 'adam'
bidirectional = False
metrics = [MeanAbsoluteError()]

# Adjust model creation to include hyperparameters
input_shape = (X_train.shape[1], n_features)

# Create the model
model = create_dl_model(
    input_shape=input_shape,
    n_layers=n_layers,
    units=units,
    layer_names=layer_names,
    dropout_rate=dropout_rate,
    loss=loss,
    optimizer=optimizer,
    bidirectional=bidirectional,
    metrics=metrics
)

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# ------------------------------------------MODEL SAVING AND LOADING---------------------------------------------------
# Define the model's filename for saving/loading
model_filename = f"{data_directory}/{COMPANY}_model.h5"

if os.path.exists(model_filename):
    model = tf.keras.models.load_model(model_filename)
else:
    # Create the model again if not loaded
    model = create_dl_model(
        input_shape=input_shape,
        n_layers=n_layers,
        units=units,
        layer_names=layer_names,
        dropout_rate=dropout_rate,
        loss=loss,
        optimizer=optimizer,
        bidirectional=bidirectional,
        metrics=metrics
    )
    model.fit(X_train, y_train, epochs=25, batch_size=32)
    model.save(model_filename)

# ------------------------------------------TESTING THE MODEL-----------------------------------------------------------
# Inverse transform y_test for plotting
actual_prices_inversed = scalers['Close'].inverse_transform(y_test.reshape(-1, 1)).flatten()

# Get the last sequence from the test data
last_sequence = X_test[-1]

# Predict using multistep prediction (e.g., for the next 5 days)
future_prices_multistep = multistep_prediction(model, last_sequence, scalers, days=5)
print(f"Multistep predicted prices for the next 5 days: {future_prices_multistep}")

# Predict using multivariate prediction (for one specific day)
predicted_price_multivariate = multivariate_prediction(model, X_test, scalers)
print(f"Multivariate predicted closing price: {predicted_price_multivariate}")

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(range(len(actual_prices_inversed)), actual_prices_inversed, color="black", label="Actual Prices")

# Plot multistep predictions
plt.plot(range(len(actual_prices_inversed), len(actual_prices_inversed) + len(future_prices_multistep)),
         future_prices_multistep, color="blue", label="Multistep Predicted Prices")

# Plot multivariate prediction as a single point
plt.scatter(len(actual_prices_inversed), predicted_price_multivariate, color="red", label="Multivariate Predicted Price")

plt.title(f"{COMPANY} Share Price Predictions")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# --------------------------------------------CANDLESTICK CHART---------------------------------------------------------
def plot_candlestick(input_df, n=1):
    """
    This function plots a candlestick chart for the given stock data.

    Parameters:
    - input_df: DataFrame, the stock data to plot.
    - n: int, number of days to aggregate into a single candlestick (default is 1).

     Explanation:
    - When n=1, each candlestick represents one trading day.
    - When n>1, the data is resampled over n trading days, where:
        * 'Open' is the first opening price within the n days.
        * 'High' is the maximum price within the n days.
        * 'Low' is the minimum price within the n days.
        * 'Close' is the last closing price within the n days.
        * 'Volume' is the sum of volumes over the n days.
    """
    # Resampling the data to group by n trading days
    if n > 1:
        input_df = input_df.resample(f'{n}D').agg({
            'Open': 'first',  # First opening price in the n-day window
            'High': 'max',      # Maximum price in the n-day window
            'Low': 'min',        # Minimum price in the n-day window
            'Close': 'last',    # Last closing price in the n-day window
            'Volume': 'sum'      # Sum of volumes in the n-day window
        }).dropna()

    # Plotting the candlestick chart using mplfinance
    mpf.plot(input_df, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', volume=True)

# ---------------------------------------------BOXPLOT CHART------------------------------------------------------------
def plot_boxplot(input_df, n=1, k=10):
    """
    This function plots a boxplot for the stock data.

    Parameters:
    - input_df: DataFrame, the stock data to plot.
    - n: int, number of days to aggregate into a single period (default is 1, i.e., no aggregation).
    - k: int, step size for x-axis labels (default is 10).

    Explanation:
    - This function resamples the data for n trading days if n > 1.
    - The boxplot visualises the distribution of 'Low', 'Open', 'Close', and 'High' prices over the period.
    - The x-axis labels are adjusted for clarity based on the value of k.
    """
    # Copying to avoid warnings
    input_df = input_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        input_df = input_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

    # Preparing the data for boxplot
    box_data = []
    labels = []
    for idx, row in input_df.iterrows():
        box_data.append([row['Low'], row['Open'], row['Close'], row['High']])
        labels.append(idx.strftime('%Y-%m-%d'))

    # Plotting
    fig, ax = plt.subplots()
    ax.boxplot(box_data, vert=True, patch_artist=True)
    ax.set_title(f'{COMPANY} Boxplot Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Set x-axis labels and ticks
    ax.set_xticks(range(1, len(labels) + 1, k))
    ax.set_xticklabels(labels[::k], rotation=90)

    plt.show()

# Now call the functions with test_data
# Download the test dataset to evaluate the model's performance.
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'
test_data = yf.download(COMPANY, TEST_START, TEST_END)
actual_prices = test_data['Close'].values

plot_candlestick(test_data, n=5)
plot_boxplot(test_data, n=1, k=10)

# --------------------------------------------ADDITIONAL CHARTS & VISUALS-----------------------------------------------
# Candlestick chart for the test period using mplfinance
mpf.plot(test_data, type='candle', style='charles', title=f'{COMPANY} Candlestick Chart', ylabel='Price', volume=True)

# Plotting high and low prices for the test period.
plt.figure(figsize=(14, 7))
plt.plot(test_data['High'], color="blue", label="High Price")
plt.plot(test_data['Low'], color="red", label="Low Price")
plt.title(f"{COMPANY} High and Low Prices")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()

# -------------------------------------------PREDICTING FUTURE DAYS-----------------------------------------------------
# Predicting the next 5 days using the trained model.
num_days_to_predict = 5
future_predictions = []

# Prepare the combined dataset for testing, including both training and new test data.
total_dataset = pd.concat(
    (pd.DataFrame(y_train, columns=['Close']), test_data[['Open', 'High', 'Low', 'Close', 'Volume']]), axis=0)

# Apply the stored scalers to the features in the combined dataset.
for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
    total_dataset[feature] = scalers[feature].transform(total_dataset[feature].values.reshape(-1, 1))

# Extract the model input sequences from the total dataset.
model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values

for _ in range(num_days_to_predict):
    # Prepare the latest data sequence for prediction.
    real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:]]

    real_data = np.array(real_data)
    prediction = model.predict(real_data)

    future_predictions.append(prediction[0, 0])
    model_inputs = np.append(model_inputs, [[*model_inputs[-1][1:], prediction[0, 0]]], axis=0)

# Inverse transform the predictions to get them back to the original price scale.
future_predictions = scalers['Close'].inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot the predicted future prices for the next 5 days.
plt.figure(figsize=(14, 7))
plt.plot(range(len(actual_prices)), actual_prices, color="black", label="Actual Prices")
plt.plot(range(len(actual_prices), len(actual_prices) + num_days_to_predict), future_predictions.flatten(), color="blue",
         label="Predicted Next Days")
plt.title(f"{COMPANY} Share Price Prediction for Next {num_days_to_predict} Days")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Price")
plt.legend()
plt.show()
