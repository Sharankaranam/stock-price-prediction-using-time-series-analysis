# stock-price-prediction-using-time-series-analysis

This GitHub repository contains a web application built with Flask that predicts stock prices for a given stock symbol using a Long Short-Term Memory (LSTM) neural network. Users can input a stock symbol, and the application fetches historical stock price data from Alpha Vantage, preprocesses it, and uses an LSTM model for prediction. The predicted prices are displayed alongside the actual historical prices in an interactive web interface.

Key Features:

Fetches historical stock price data from Alpha Vantage API.
Utilizes Min-Max scaling to normalize the data for training.
Implements an LSTM model for stock price prediction.
Allows users to input a stock symbol and view predictions.
Provides an interactive web interface for easy access.

Dependencies:
Python 3.x
Flask
Alpha Vantage Python Library
scikit-learn
TensorFlow

Note: Ensure you have the required dependencies installed and properly configured before running the application.
