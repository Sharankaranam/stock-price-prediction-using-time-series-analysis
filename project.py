import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Define API key
api_key = 'B9I2SS3DP9ZKFC9Y'

# stock symbol and time period
symbol = input("Enter the stock symbol (e.g., AAPL): ").strip().upper()
interval = '1d'  # Daily data

# formatting the data
ts = TimeSeries(key=api_key, output_format='pandas')

# Fetch historical stock price data
data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data = data.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
data = data.sort_index(ascending=True)

# Extract the prices for prediction
prices = data['Close'].values
prices = prices.reshape(-1, 1)

# Normalize & using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(prices)

# Splitting the data
train_size = int(len(prices_normalized) * 0.8)
train_data = prices_normalized[:train_size]
test_data = prices_normalized[train_size:]

# LSTM 
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10 
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the predictions to original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate RMSE (Root Mean Squared Error) on the test set
test_rmse = np.sqrt(mean_squared_error(prices[-len(test_predictions):], test_predictions))
print(f'Test RMSE: {test_rmse}')

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(prices[-len(test_predictions):], label='True Prices')
plt.plot(test_predictions, label='Predicted Prices')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{symbol} Stock Price Prediction (Test Data)')
plt.show()