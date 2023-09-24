import requests
import pandas as pd
from flask import Flask, render_template, request

# Define your Alpha Vantage API key
api_key = 'YOUR_API_KEY_HERE'

# Initialize Flask
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the stock symbol entered by the user
        symbol = request.form['symbol'].strip().upper()

        # Define the API endpoint URL with the user-provided symbol
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY' \
                  f'&symbol={symbol}&interval=1min&apikey={api_key}'

        try:
            # Make a GET request to the Alpha Vantage API
            response = requests.get(api_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the JSON response data
                data = response.json()

                # Check if the 'Time Series (1min)' key exists in the response data
                if 'Time Series (1min)' in data:
                    # Extract the intraday stock price data
                    df = pd.DataFrame.from_dict(data['Time Series (1min)'], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df['4. close'] = df['4. close'].astype(float)

                    # Calculate the Simple Moving Average (SMA) with a 10-minute window
                    sma_window = 10
                    df['SMA'] = df['4. close'].rolling(window=sma_window).mean()

                    # Get the last calculated SMA value
                    last_sma = df['SMA'].iloc[-1]

                    return render_template('index.html', symbol=symbol, last_sma=last_sma)
                else:
                    return render_template('index.html', error=f"No data found for {symbol}.")
            else:
                return render_template('index.html', error=f"API request failed with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            return render_template('index.html', error=f"Error: {e}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
