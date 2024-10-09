import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

def fetch_stock_data(ticker):
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="1d", interval="1m")  # Fetch 1-minute interval data for the current day
    return hist_data

def preprocess_data(data):
    data['Datetime'] = data.index
    data['Hour'] = data['Datetime'].dt.hour
    data['Minute'] = data['Datetime'].dt.minute
    data['Time'] = data['Hour'] + data['Minute'] / 60.0
    
    # Calculate SMA, EMA, RSI, and MACD
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data['MACD'], data['Signal Line'], _ = compute_macd(data['Close'], 12, 26, 9)
    
    data = data.dropna()  # Drop rows with NaN values
    data = data[['Time', 'Close', 'Open', 'SMA_5', 'EMA_5', 'RSI', 'MACD']]
    return data

def compute_rsi(close, window):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close, short_window, long_window, signal_window):
    short_ema = close.ewm(span=short_window, adjust=False).mean()
    long_ema = close.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def train_models(data):
    X = np.array(data[['Time', 'SMA_5', 'EMA_5', 'RSI', 'MACD']])
    y = np.array(data['Close'])
    
    lr_model = LinearRegression()
    dt_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    lr_model.fit(X, y)
    dt_model.fit(X, y)
    rf_model.fit(X, y)
    
    return lr_model, dt_model, rf_model

def make_ensemble_predictions(models, data):
    times = np.arange(9.1667, 15.5, 1/60)  # 9:10 AM to 3:30 PM in fractional hours
    sma_5 = pd.Series(data['Close']).rolling(window=5).mean().dropna().values
    ema_5 = pd.Series(data['Close']).ewm(span=5, adjust=False).mean().dropna().values
    rsi = compute_rsi(data['Close'], window=14).dropna().values
    macd, _, _ = compute_macd(data['Close'], 12, 26, 9)
    
    min_length = min(len(times), len(sma_5), len(ema_5), len(rsi), len(macd))
    
    if min_length == 0:
        return np.array([]), np.array([])  # Return empty arrays if no valid data points

    features = np.column_stack((times[:min_length], sma_5[:min_length], ema_5[:min_length], rsi[:min_length], macd[:min_length]))
    
    lr_predictions = models[0].predict(features)
    dt_predictions = models[1].predict(features)
    rf_predictions = models[2].predict(features)
    
    ensemble_predictions = (lr_predictions + dt_predictions + rf_predictions) / 3
    return times[:min_length], ensemble_predictions

def plot_predictions(times, prices, ticker):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times[:len(prices)], prices, label='Predicted Prices', color='blue', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(f'Hourly Predictions for {ticker}')
    ax.legend()
    plt.xticks(rotation=45)

    # Setting x-ticks to show hourly intervals more clearly
    time_labels = [f"{int(9 + i // 60)}:{int(i % 60):02d}" for i in range(0, len(times), 60)]
    ax.set_xticks(times[::60])
    ax.set_xticklabels(time_labels)
    
    return fig

def fetch_real_closing_price(ticker):
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="1d")  # Fetch daily data
    closing_price = hist_data['Close'][-1]  # Get the latest closing price
    return closing_price

def fetch_current_price(ticker):
    stock_data = yf.Ticker(ticker)
    current_price = stock_data.history(period="1d", interval="1m")['Close'][-1]  # Get the latest price
    return current_price

# Streamlit GUI
st.title('Stock Market Price Prediction')
ticker = st.text_input('Enter Stock Ticker:', 'RELIANCE.NS')

if st.button('Predict'):
    data = fetch_stock_data(ticker)
    if not data.empty:
        open_price = data['Open'].iloc[0]  # Fetch the open price of the stock
        st.write(f"Open price of {ticker}: {open_price}")

        processed_data = preprocess_data(data)
        if not processed_data.empty:
            models = train_models(processed_data)
            times, predictions = make_ensemble_predictions(models, processed_data)

            if len(times) > 0:
                closing_price_prediction = predictions[-1]
                st.write(f"Predicted closing price: {closing_price_prediction}")

                fig = plot_predictions(times, predictions, ticker)
                st.pyplot(fig)

                # Display additional features
                st.write("Additional Features:")
                st.write(f"- SMA_5 (5-Day Simple Moving Average): {processed_data['SMA_5'].iloc[-1]}")
                st.write(f"- EMA_5 (5-Day Exponential Moving Average): {processed_data['EMA_5'].iloc[-1]}")
                st.write(f"- RSI (Relative Strength Index): {processed_data['RSI'].iloc[-1]}")
                st.write(f"- MACD (Moving Average Convergence Divergence): {processed_data['MACD'].iloc[-1]}")

                current_time = datetime.now().time()
                market_close_time = datetime.strptime("15:30", "%H:%M").time()

                if current_time < market_close_time:
                    real_or_current_price = fetch_current_price(ticker)
                    st.write(f"Current market price: {real_or_current_price}")
                else:
                    real_or_current_price = fetch_real_closing_price(ticker)
                    st.write(f"Real closing price: {real_or_current_price}")

                mae = mean_absolute_error([real_or_current_price], [closing_price_prediction])
                mse = mean_squared_error([real_or_current_price], [closing_price_prediction])

                st.write(f"Mean Absolute Error (MAE): {mae}")
                st.write(f"Mean Squared Error (MSE): {mse}")
            else:
                st.write('Not enough data points to make predictions.')
        else:
            st.write('No valid data after preprocessing.')
    else:
        st.write('No data available for the provided ticker.')  