# COS30018 - Intelligent Systems
# Task B.6 - Machine Learning 3
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
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor  # For Random Forest model
import pmdarima as pm  # For SARIMA model

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
        '~/Desktop/COS30018 - Intelligent Systems/Assignment_B/Task B.6 - Machine Learning 3')
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

    # Save a copy of the original data before scaling for models that require unscaled data.
    unscaled_data = data.copy()

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

        # Corresponding unscaled data
        unscaled_train_data = unscaled_data.iloc[:train_size]
        unscaled_test_data = unscaled_data.iloc[train_size:]
    else:
        train_data, test_data, unscaled_train_data, unscaled_test_data = train_test_split(
            data, unscaled_data, test_size=1 - split_ratio, random_state=random_state, shuffle=True)

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
    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
            scalers, data_directory, data, unscaled_train_data, unscaled_test_data)

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
        num_layers = config['n_layers']
        units = config['units']
        layer_names = config['layer_names']
        dropout_rate = config.get('dropout_rate', 0.2)  # Default dropout rate is 0.2 if not provided.
        epochs = config['epochs']
        batch_size = config['batch_size']
        bidirectional = config.get('bidirectional', False)
        loss = config.get('loss', 'mean_absolute_error')
        optimizer = config.get('optimizer', 'adam')
        metrics = config.get('metrics', [MeanAbsoluteError()])

        # Create the model using the extracted parameters.
        input_shape = (X_train.shape[1], X_train.shape[2])  # Input shape based on the number of past days and features.
        model = create_dl_model(
            input_shape=input_shape,
            n_layers=num_layers,
            units=units,
            layer_names=layer_names,
            dropout_rate=dropout_rate,
            loss=loss,
            optimizer=optimizer,
            bidirectional=bidirectional,
            metrics=metrics
        )

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
    Predicts multiple future steps ahead using the last available data sequence.

    Parameters:
    - model: Trained deep learning model.
    - last_sequence: The last sequence from the test data.
    - scalers: Dictionary of scalers used to inverse transform the prediction.
    - days: The number of future days to predict.

    Returns:
    - future_predictions: List of predicted closing prices for 'days' future days.
    """
    # Plot validation loss across epochs for each configuration.
    plt.figure(figsize=(10, 6))
    for idx, history in enumerate(validation_histories):
        plt.plot(history,
                 label=f"Config {idx + 1}: {results[idx]['config']['layer_names']} - Layers: {results[idx]['config']['n_layers']}")
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
(X_train, y_train, X_test, y_test, scalers, data_directory, data,
 unscaled_train_data, unscaled_test_data) = load_and_process_data(
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

# --------------------------------------EXPERIMENTING WITH HYPERPARAMETERS (LSTSM & GRU)--------------------------------
# Define hyperparameter configurations
hyperparameter_configurations = [
    {
        'n_layers': 2,
        'units': [50, 50],
        'layer_names': ['LSTM', 'LSTM'],
        'dropout_rate': 0.2,
        'epochs': 10,
        'batch_size': 32,
        'bidirectional': False
    },
    {
        'n_layers': 3,
        'units': [64, 64, 64],
        'layer_names': ['GRU', 'GRU', 'GRU'],
        'dropout_rate': 0.3,
        'epochs': 10,
        'batch_size': 32,
        'bidirectional': False
    },
    {
        'n_layers': 2,
        'units': [50, 50],
        'layer_names': ['LSTM', 'LSTM'],
        'dropout_rate': 0.2,
        'epochs': 10,
        'batch_size': 32,
        'bidirectional': True
    },
]

# Run experiments
results, validation_histories = experiment_with_hyperparameters(X_train, y_train, X_test, y_test, hyperparameter_configurations)

# Visualise results
plot_validation_loss(results, validation_histories)

# Choose the best model based on validation loss
best_model_config = min(results, key=lambda x: x['val_loss'])
print(f"Best model configuration: {best_model_config['config']} with validation loss: {best_model_config['val_loss']}")

# Create and train the best model
best_config = best_model_config['config']
input_shape = (X_train.shape[1], X_train.shape[2])
best_model = create_dl_model(
    input_shape=input_shape,
    n_layers=best_config['n_layers'],
    units=best_config['units'],
    layer_names=best_config['layer_names'],
    dropout_rate=best_config.get('dropout_rate', 0.2),
    loss=best_config.get('loss', 'mean_absolute_error'),
    optimizer=best_config.get('optimizer', 'adam'),
    bidirectional=best_config.get('bidirectional', False),
    metrics=best_config.get('metrics', [MeanAbsoluteError()])
)
best_model.fit(X_train, y_train, epochs=best_config['epochs'], batch_size=best_config['batch_size'],
               validation_data=(X_test, y_test), verbose=1)

# Save the best model
model_filename = f"{data_directory}/{COMPANY}_best_model.h5"
best_model.save(model_filename)

# ------------------------------------------RANDOM FOREST MODEL--------------------------------------------------------
# Prepare data for Random Forest using unscaled data
def create_rf_features(data, sequence_length):
    X_rf = []
    y_rf = []
    for i in range(sequence_length, len(data)):
        X_rf.append(data.iloc[i-sequence_length:i].values.flatten())
        y_rf.append(data.iloc[i]['Close'])
    return np.array(X_rf), np.array(y_rf)

# Create training and testing data for Random Forest
X_train_rf, y_train_rf = create_rf_features(unscaled_train_data, PREDICTION_DAYS)
X_test_rf, y_test_rf = create_rf_features(unscaled_test_data, PREDICTION_DAYS)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Make predictions with Random Forest
predictions_rf = rf_model.predict(X_test_rf)
# -------------------------------------------------SARIMA MODEL --------------------------------------------------------
# Prepare data for SARIMA using unscaled data
train_data_sarima = unscaled_train_data['Close']
test_data_sarima = unscaled_test_data['Close']

# Fit SARIMA model
sarima_model = pm.auto_arima(
    train_data_sarima,
    start_p=1, start_q=1,
    max_p=5, max_q=5,
    d=None,  # Differencing will be determined automatically
    seasonal=True,  # SARIMA instead of ARIMA
    m=12,  # Set seasonality period. For monthly seasonality, it's 12; adjust accordingly.
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=1,  # Seasonal differencing
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)
print(sarima_model.summary())

# Make predictions with SARIMA
n_periods = len(test_data_sarima)
predictions_sarima = sarima_model.predict(n_periods=n_periods)
predictions_sarima = np.array(predictions_sarima)

# ------------------------------------------COMBINING PREDICTIONS-------------------------------------------------------
# Make predictions with the best DL model on X_test
predictions_dl = best_model.predict(X_test)
# Inverse transform the predictions
predictions_dl_inversed = scalers['Close'].inverse_transform(predictions_dl).flatten()

# Inverse transform y_test for comparison
y_test_inversed = scalers['Close'].inverse_transform(y_test.reshape(-1, 1)).flatten()

# Ensure the lengths match
min_len = min(len(predictions_sarima), len(predictions_dl_inversed), len(predictions_rf), len(y_test_inversed))
predictions_sarima = predictions_sarima[:min_len]
predictions_dl_inversed = predictions_dl_inversed[:min_len]
predictions_rf = predictions_rf[:min_len]
y_test_inversed = y_test_inversed[:min_len]

# Combine predictions (ensemble of SARIMA, DL, and RF)
combined_predictions = (predictions_sarima + predictions_dl_inversed + predictions_rf) / 3

# ------------------------------------------EVALUATING THE MODELS-------------------------------------------------------
# Evaluate models
mae_sarima = mean_absolute_error(y_test_inversed, predictions_sarima)
mae_dl = mean_absolute_error(y_test_inversed, predictions_dl_inversed)
mae_rf = mean_absolute_error(y_test_inversed, predictions_rf)
mae_combined = mean_absolute_error(y_test_inversed, combined_predictions)

rmse_sarima = np.sqrt(mean_squared_error(y_test_inversed, predictions_sarima))
rmse_dl = np.sqrt(mean_squared_error(y_test_inversed, predictions_dl_inversed))
rmse_rf = np.sqrt(mean_squared_error(y_test_inversed, predictions_rf))
rmse_combined = np.sqrt(mean_squared_error(y_test_inversed, combined_predictions))

print("MAE SARIMA:", mae_sarima)
print("MAE Deep Learning:", mae_dl)
print("MAE Random Forest:", mae_rf)
print("MAE Combined:", mae_combined)

print("RMSE SARIMA:", rmse_sarima)
print("RMSE Deep Learning:", rmse_dl)
print("RMSE Random Forest:", rmse_rf)
print("RMSE Combined:", rmse_combined)

# ------------------------------------------VISUALISING THE RESULTS-----------------------------------------------------
# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_inversed, label='Actual Prices', color='black')
plt.plot(predictions_sarima, label='SARIMA Predictions', color='green')
plt.plot(predictions_dl_inversed, label='DL Predictions', color='blue')
plt.plot(predictions_rf, label='RF Predictions', color='orange')
plt.plot(combined_predictions, label='Combined Predictions', color='red')
plt.title(f"{COMPANY} Stock Price Predictions")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
# --------------------------------------------CANDLESTICK CHART---------------------------------------------------------
def plot_candlestick(input_df, n=1):
    """
    [Function documentation remains the same]
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
    [Function documentation remains the same]
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
    prediction = best_model.predict(real_data)

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
