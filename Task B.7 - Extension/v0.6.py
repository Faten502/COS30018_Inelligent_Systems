# COS30018 - Intelligent Systems
# Task B.7 - Extension
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
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For sentiment analysis

# --------------------------------------NEWS API & SENTIMENT ANALYSIS------------------------------------------------
def fetch_news_sentiment(company_name, api_key):
    """
    Fetch news articles related to the company and calculate sentiment scores using VADER.
    """
    # Construct API URL to fetch news articles related to the company
    url = f'https://newsapi.org/v2/everything?q={company_name}&apiKey={api_key}'

    # Fetch articles from News API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching news data: {response.status_code} - {response.reason}")
        return None  # If there's an error, proceed without sentiment data

    # Parse the articles from the response JSON
    articles = response.json().get('articles', [])
    analyzer = SentimentIntensityAnalyzer()  # Initialise VADER sentiment analyser

    # Initialise list to hold sentiment scores for each article's headline
    headlines_sentiment = []
    for article in articles:
        headline = article['title']  # Extract headline
        published_at = article['publishedAt']  # Extract publication date
        # Calculate sentiment score using VADER (compound score)
        sentiment_score = analyzer.polarity_scores(headline)['compound']
        # Store date, headline, and sentiment score
        headlines_sentiment.append({'Date': published_at, 'Headline': headline, 'Sentiment_Score': sentiment_score})

    # Convert the list to a DataFrame for easier manipulation
    headlines_df = pd.DataFrame(headlines_sentiment)

    # Convert the 'Date' column to pandas datetime format and group by date to get daily sentiment scores
    headlines_df['Date'] = pd.to_datetime(headlines_df['Date']).dt.date
    daily_sentiment = headlines_df.groupby('Date')['Sentiment_Score'].mean()

    return daily_sentiment  # Return the daily sentiment scores
# --------------------------------------FUNCTION FOR LOADING & PROCESSING WITH SENTIMENT--------------------------------
def load_and_process_data_with_sentiment(ticker, start_date, end_date, features=['Open', 'High', 'Low', 'Close', 'Volume'],
                                         sentiment_data=None, handle_nan=True, nan_strategy='drop', split_ratio=0.8,
                                         scale_features=True, PREDICTION_DAYS=60):
    """
    Function to load stock data, merge it with sentiment data, and process it for model training.

    Parameters:
    - ticker: Stock symbol (e.g., 'CBA.AX')
    - start_date: Start date for stock data
    - end_date: End date for stock data
    - features: List of stock features to include (default: ['Open', 'High', 'Low', 'Close', 'Volume'])
    - sentiment_data: Dictionary of sentiment data (dates and sentiment scores)
    - handle_nan: Whether to handle missing data (default: True)
    - nan_strategy: Strategy for handling NaN values ('drop' or 'fill')
    - split_ratio: Ratio for train/test split (default: 0.8)
    - scale_features: Whether to scale features to the range [0, 1]
    - PREDICTION_DAYS: Number of past days to use for prediction

    Returns:
    - X_train, y_train: Training feature and target sets
    - X_test, y_test: Testing feature and target sets
    - scalers: Dictionary of fitted scalers for inverse transformation
    - data: Processed and scaled stock data
    - unscaled_data: Original unscaled stock data
    """
    # Download stock data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Handle missing data if necessary
    if handle_nan:
        if nan_strategy == 'drop':
            data.dropna(inplace=True)
        elif nan_strategy == 'fill':
            data.fillna(method='ffill', inplace=True)

    # Convert all data to float type
    data = data[features].astype(float)

    # Merge sentiment data with stock price data if available
    if sentiment_data is not None:
        data['Sentiment'] = data.index.map(lambda x: sentiment_data.get(x.date(), 0))
        # Avoid chained assignment warning
        data['Sentiment'] = data['Sentiment'].fillna(0)  # Handle missing sentiment data
        features_with_sentiment = features + ['Sentiment']
    else:
        features_with_sentiment = features

    # Save a copy of the original data before scaling for future use
    unscaled_data = data.copy()

    # Scale features if required
    scalers = {}
    if scale_features:
        for feature in features_with_sentiment:
            scaler = MinMaxScaler()
            data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            scalers[feature] = scaler

    # Print the total number of data points before splitting
    print(f"Total data points: {len(data)}")

    # Split data into training and testing sets
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")

    # Prepare data for model input
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(PREDICTION_DAYS, len(train_data)):
        X_train.append(train_data.iloc[i - PREDICTION_DAYS:i][features_with_sentiment].values)
        y_train.append(train_data.iloc[i]['Close'])

    for i in range(PREDICTION_DAYS, len(test_data)):
        X_test.append(test_data.iloc[i - PREDICTION_DAYS:i][features_with_sentiment].values)
        y_test.append(test_data.iloc[i]['Close'])

    # Convert the sequences to numpy arrays
    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test),
            scalers, data, unscaled_data, train_data, test_data)
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
    - dropout_rate: float, dropout rate for regularisation (default 0.3).
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
    Plots validation loss per epoch for each hyperparameter configuration.

    Parameters:
    - results: List of dictionaries containing configurations and their validation losses.
    - validation_histories: List of validation loss histories per configuration.
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
TRAIN_START = '2023-01-01'  # Adjusted to include more data
TRAIN_END = '2023-10-01'    # End date remains the same
PREDICTION_DAYS = 5         # Adjusted to fit the available data

# Fetch sentiment data for Commonwealth Bank
API_KEY = '5bf9f4d9f01f464d9cc9bd7d7997130e'
sentiment_data = fetch_news_sentiment('Commonwealth Bank', API_KEY)

# Load, process, and split the stock data with sentiment
(X_train_with_senti, y_train_with_senti, X_test_with_senti, y_test_with_senti, scalers_with_senti, data_with_senti,
 unscaled_data_with_senti, unscaled_train_data_with_senti, unscaled_test_data_with_senti) = load_and_process_data_with_sentiment(
    COMPANY, TRAIN_START, TRAIN_END, features=['Open', 'High', 'Low', 'Close', 'Volume'],
    sentiment_data=sentiment_data, split_ratio=0.7, scale_features=True, PREDICTION_DAYS=PREDICTION_DAYS
)

# Load, process, and split the stock data without sentiment
(X_train_no_senti, y_train_no_senti, X_test_no_senti, y_test_no_senti, scalers_no_senti, data_no_senti,
 unscaled_data_no_senti, unscaled_train_data_no_senti, unscaled_test_data_no_senti) = load_and_process_data_with_sentiment(
    COMPANY, TRAIN_START, TRAIN_END, features=['Open', 'High', 'Low', 'Close', 'Volume'],
    sentiment_data=None, split_ratio=0.7, scale_features=True, PREDICTION_DAYS=PREDICTION_DAYS
)

# Verify the shapes
print(f"With Sentiment - X_train shape: {X_train_with_senti.shape}, X_test shape: {X_test_with_senti.shape}")
print(f"Without Sentiment - X_train shape: {X_train_no_senti.shape}, X_test shape: {X_test_no_senti.shape}")

# Determine the number of features for both datasets
n_features_with_senti = len(['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment'])
n_features_no_senti = len(['Open', 'High', 'Low', 'Close', 'Volume'])

# Reshape the data to fit the input shape expected by the models
X_train_with_senti = np.reshape(X_train_with_senti, (X_train_with_senti.shape[0], X_train_with_senti.shape[1], n_features_with_senti))
X_test_with_senti = np.reshape(X_test_with_senti, (X_test_with_senti.shape[0], X_test_with_senti.shape[1], n_features_with_senti))

X_train_no_senti = np.reshape(X_train_no_senti, (X_train_no_senti.shape[0], X_train_no_senti.shape[1], n_features_no_senti))
X_test_no_senti = np.reshape(X_test_no_senti, (X_test_no_senti.shape[0], X_test_no_senti.shape[1], n_features_no_senti))

# ------------------------------------------MODEL SAVING AND LOADING---------------------------------------------------
# Define the model filenames
model_filename_with_senti = f"{COMPANY}_model_with_sentiment_{PREDICTION_DAYS}days.h5"
model_filename_no_senti = f"{COMPANY}_model_without_sentiment_{PREDICTION_DAYS}days.h5"

# Model with sentiment
if os.path.exists(model_filename_with_senti):
    model_with_senti = tf.keras.models.load_model(model_filename_with_senti)
else:
    # Create and compile the model
    input_shape_with_senti = (X_train_with_senti.shape[1], X_train_with_senti.shape[2])
    model_with_senti = create_dl_model(
        input_shape=input_shape_with_senti,
        n_layers=3, units=[50, 100, 150], layer_names=['LSTM', 'GRU', 'RNN'],
        dropout_rate=0.3, loss='mean_absolute_error', optimizer='adam'
    )
    model_with_senti.fit(X_train_with_senti, y_train_with_senti, epochs=25, batch_size=32, validation_data=(X_test_with_senti, y_test_with_senti))
    model_with_senti.save(model_filename_with_senti)

# Model without sentiment
if os.path.exists(model_filename_no_senti):
    model_no_senti = tf.keras.models.load_model(model_filename_no_senti)
else:
    # Create and compile the model
    input_shape_no_senti = (X_train_no_senti.shape[1], X_train_no_senti.shape[2])
    model_no_senti = create_dl_model(
        input_shape=input_shape_no_senti,
        n_layers=3, units=[50, 100, 150], layer_names=['LSTM', 'GRU', 'RNN'],
        dropout_rate=0.3, loss='mean_absolute_error', optimizer='adam'
    )
    model_no_senti.fit(X_train_no_senti, y_train_no_senti, epochs=25, batch_size=32, validation_data=(X_test_no_senti, y_test_no_senti))
    model_no_senti.save(model_filename_no_senti)
# ------------------------------------------TESTING THE MODELS-----------------------------------------------------------
# Inverse transform y_test for plotting
actual_prices_inversed_with_senti = scalers_with_senti['Close'].inverse_transform(y_test_with_senti.reshape(-1, 1)).flatten()
actual_prices_inversed_no_senti = scalers_no_senti['Close'].inverse_transform(y_test_no_senti.reshape(-1, 1)).flatten()

# Predictions with model with sentiment
predictions_with_senti = model_with_senti.predict(X_test_with_senti)
predictions_with_senti_inversed = scalers_with_senti['Close'].inverse_transform(predictions_with_senti).flatten()

# Predictions with model without sentiment
predictions_no_senti = model_no_senti.predict(X_test_no_senti)
predictions_no_senti_inversed = scalers_no_senti['Close'].inverse_transform(predictions_no_senti).flatten()

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(actual_prices_inversed_with_senti, color="black", label="Actual Prices")
plt.plot(predictions_with_senti_inversed, color="blue", label="Predictions with Sentiment")
plt.plot(predictions_no_senti_inversed, color="green", label="Predictions without Sentiment")
plt.title(f"{COMPANY} Share Price Predictions with and without Sentiment")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Evaluate models
mae_with_senti = mean_absolute_error(actual_prices_inversed_with_senti, predictions_with_senti_inversed)
mae_no_senti = mean_absolute_error(actual_prices_inversed_no_senti, predictions_no_senti_inversed)

rmse_with_senti = np.sqrt(mean_squared_error(actual_prices_inversed_with_senti, predictions_with_senti_inversed))
rmse_no_senti = np.sqrt(mean_squared_error(actual_prices_inversed_no_senti, predictions_no_senti_inversed))

print("MAE with Sentiment:", mae_with_senti)
print("MAE without Sentiment:", mae_no_senti)

print("RMSE with Sentiment:", rmse_with_senti)
print("RMSE without Sentiment:", rmse_no_senti)

# --------------------------------------EXPERIMENTING WITH HYPERPARAMETERS (LSTM & GRU)--------------------------------
# Before running experiments, define which dataset to use
# Let's use the dataset with sentiment data
X_train = X_train_with_senti
y_train = y_train_with_senti
X_test = X_test_with_senti
y_test = y_test_with_senti
scalers = scalers_with_senti
unscaled_train_data = unscaled_train_data_with_senti  # Add this line
unscaled_test_data = unscaled_test_data_with_senti    # Add this line

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
best_model_filename = f"{COMPANY}_best_model_{PREDICTION_DAYS}days.h5"
best_model.save(best_model_filename)

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
TEST_END = '2023-10-01'
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
