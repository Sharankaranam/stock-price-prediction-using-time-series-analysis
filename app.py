import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define your Alpha Vantage API key
api_key = 'YOUR_API_KEY_HERE'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the stock symbol entered by the user
        symbol = request.form['symbol'].strip().upper()

        # Fetch stock price data and make predictions
        prices, test_predictions = predict_stock_price(symbol)

        if prices is not None and test_predictions is not None:
            return render_template('index.html', symbol=symbol, prices=prices.tolist(), predictions=test_predictions.tolist())

    return render_template('index.html', symbol=None, prices=None, predictions=None)

def predict_stock_price(symbol, seq_length=10):
    try:
        # Initialize Alpha Vantage API client
        ts = TimeSeries(key=api_key, output_format='pandas')

        # Fetch historical stock price data
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
        data = data.sort_index(ascending=True)

        # Extract the 'Close' prices for prediction
        prices = data['Close'].values
        prices = prices.reshape(-1, 1)

        # Normalize the data using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_normalized = scaler.fit_transform(prices)

        # Split the data into training and testing sets
        train_size = int(len(prices_normalized) * 0.8)
        train_data = prices_normalized[:train_size]
        test_data = prices_normalized[train_size:]

        # Create sequences for LSTM training
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)

        # Build an LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(seq_length, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        # Inverse transform the predictions to original scale
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)

        return prices[-len(test_predictions):], test_predictions

    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == '__main__':
    app.run(debug=True)
