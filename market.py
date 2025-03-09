import numpy as np
import connection
import csv
import os
import time

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
    current_candle_bids = []  # List to store 3 bid prices

    for _ in range(5):  # Collect data every minute for 5 minutes
        bid_price, timestamp = fetch_current_market_info()
        current_candle_bids.append(bid_price)
        time.sleep(60)  # Wait for the next minute

    # Calculate candle values
    open_price = current_candle_bids[0]
    close_price = sum(current_candle_bids) / 3  # Average bid as close
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

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Append to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, market_status, round(volatility, 4), fast_changes])

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
def calculate_macd_values(lis_fetched_candles):
    global lis_macd_lines
    flo_signal_line = 0.0
    # Calculate the long-term EMA (26-period)
    arr_prices = [candle[1] for candle in lis_fetched_candles[:26]]  # Get the latest 26 candles from the beginning of the list
    ema_12 = calculate_ema(arr_prices[:12], 12)  # Calculate the ema 12 with latest 12 candles
    ema_26 = calculate_ema(arr_prices, 26)  # Calculate the ema 26 with all 26 candles
    flo_macd_line = round((ema_12 - ema_26), 2)  # Calculate the macd-line
    lis_macd_lines.append(flo_macd_line)  # Add the macd-line to the list

    if len(lis_macd_lines) > 9:  # Keep only the last 9 values for the 9-period EMA calculation
        lis_macd_lines = lis_macd_lines[-9:]
        flo_signal_line = round(calculate_ema(lis_macd_lines, 9), 2)  # Calculate the signal line
        return(flo_macd_line, flo_signal_line)
    return (flo_macd_line, flo_signal_line, lis_macd_lines)