import numpy as np
import pandas as pd
import connection
import csv
import os
import time
from datetime import datetime, timedelta

# Define CSV file path
csv_filename = "./market_conditions.csv"

# Ensure the CSV file exists with headers
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Market Condition", "Volatility", "Fast Changes"])

lis_macd_lines = []
candles = []
current_candle_bids = []


def fetch_current_market_info():
    us100_snapshot = connection.fetch_us100_snapshot()
    flo_bid = float(us100_snapshot.get('bid', 1))  # Use current bid if first candle
    str_time = us100_snapshot.get('updateTime')
    arr_one_candle = [flo_bid, str_time]  # Store Open, Close, and Time
    return arr_one_candle


def fetch_new_candle():
    global timestamp
    current_candle_bids = []  # List to store 5 bid prices

    for _ in range(5):  # Collect data every minute for 5 minutes
        bid_price, timestamp = fetch_current_market_info()
        current_candle_bids.append(bid_price)
        time.sleep(60)  # Wait for the next minute

    # Calculate candle values
    open_price = current_candle_bids[0]
    close_price = sum(current_candle_bids) / 5 # Average bid as close
    high_price = max(current_candle_bids)
    low_price = min(current_candle_bids)

    # Return the new candle
    return [open_price, close_price, timestamp, high_price, low_price]



# Function to detect market conditions

def detect_market_condition(lis_fetched_candles):
    """Determine if the market is NORMAL or SLOW based on last 3 snapshots and log results."""

    from datetime import datetime  # Import inside to avoid unnecessary global dependency

    # 1. Calculate Volatility (Last 3 Snapshots)
    bids = [snap[0] for snap in lis_fetched_candles[:3]]
    volatility = (max(bids) - min(bids)) / min(bids) * 100

    # 2. Consecutive Percentage Change
    percentage_changes = [
        ((bids[i] - bids[i - 1]) / bids[i - 1]) * 100 for i in range(1, len(bids))
    ]
    fast_changes = sum(abs(change) >= 0.1 for change in percentage_changes)

    # Decision Logic
    if volatility > 0.2 and fast_changes >= 1:
        market_status = "NORMAL"
    else:
        market_status = "SLOW"

    return market_status


def calculate_ema(prices, period):
    weighting_factor = 2 / (period + 1)  # Standard weighting factor
    ema = np.zeros(len(prices))  # Initialize an array of zeros to store the EMA values
    sma = np.mean(prices[:period])  # Calculate the Simple Moving Average (SMA) for the first 'period' elements
    ema[period - 1] = sma  # Set the initial EMA value at the end of the SMA period
    for i in range(period, len(prices)):  # Calculate the EMA for the remaining prices
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (
                    1 - weighting_factor))  # Apply the EMA formula: (current price * weighting factor) + (previous EMA * (1 - weighting factor))
    return round(ema[-1], 1)  # Return the last EMA value rounded to 1 decimal place


# This function calculates the MACD values
def calculate_macd_values(lst_fetched_candles):
    df = pd.DataFrame(lst_fetched_candles, columns=['Open', 'Close', 'Timestamp', 'High', 'Low'])

    # Bereken de EMA’s zoals in calculate_indicators()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # Bereken MACD en Signaallijn
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Haal de meest recente waarden op
    macd_value = round(df.iloc[-1]['MACD'], 2)
    signal_value = round(df.iloc[-1]['Signal'], 2)

    return macd_value, signal_value