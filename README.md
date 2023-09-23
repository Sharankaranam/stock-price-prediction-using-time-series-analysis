# stock-price-prediction-using-time-series-analysis

This GitHub repository contains a web application built with Flask that predicts stock prices for a given stock symbol using a Long Short-Term Memory (LSTM) neural network. Users can input a stock symbol, and the application fetches historical stock price data from Alpha Vantage, preprocesses it, and uses an LSTM model for prediction. The predicted prices are displayed alongside the actual historical prices in an interactive web interface.

Key Features:

1)Fetches historical stock price data from Alpha Vantage API.
2)Utilizes Min-Max scaling to normalize the data for training.
3)Implements an LSTM model for stock price prediction.
4)Allows users to input a stock symbol and view predictions.
5)Provides an interactive web interface for easy access.

Dependencies:
1)Python 3.x,
2)Flask;
3)Alpha Vantage Python Library;
4)scikit-learn;
5)TensorFlow;

Note: Ensure you have the required dependencies installed and properly configured before running the application.
