import numpy as np
import connection


lis_macd_lines = []

def fetch_current_market_info():
    us100_snapshot = connection.fetch_us100_snapshot()
    # Extract bid price, netChange, percentageChange, high, and low from the snapshot
    flo_bid = float(us100_snapshot.get('bid', 1))
    flo_close = float(us100_snapshot.get('offer', 1))
    str_time = us100_snapshot.get('updateTime')
    arr_one_candle = [flo_bid, flo_close, str_time]
    return arr_one_candle


# Function to detect market conditions
def detect_market_condition(lis_fetched_candles):
    print(lis_fetched_candles[0][2])
    print(lis_fetched_candles[:3])
    """Determine if the market is NORMAL or SLOW based on last 3 snapshots."""
    # 1. Calculate Volatility (Last 3 Snapshots)
    bids = [snap[0] for snap in lis_fetched_candles[:3]]
    volatility = (max(bids) - min(bids)) / min(bids) * 100
    print("Volatility (%):", volatility)

    # 2. Calculate Spread (Last Snapshot)
    print(lis_fetched_candles[1][2])
    spread = lis_fetched_candles[1][1] - lis_fetched_candles[1][0]
    print(lis_fetched_candles[1][1], lis_fetched_candles[1][0])
    print("Spread:", spread)

    # 3. Consecutive Percentage Change
    percentage_changes = []
    for i in range(1, len(bids)):
        prev_bid = bids[i - 1]
        curr_bid = bids[i]
        percentage_change = ((curr_bid - prev_bid) / prev_bid) * 100
        percentage_changes.append(percentage_change)
        print(f"Percentage Change {i}: {percentage_change}%")

    fast_changes = sum(abs(change) >= 0.1 for change in percentage_changes)
    print("Fast Changes (>= 0.1%):", fast_changes)

    # Decision Logic
    # if volatility > 0.3 and spread < 0.01 and fast_changes >= 1:
    if volatility > 0.05 and fast_changes >= 1:
        print("NORMAL")
        return "NORMAL"
    else:
        print("SLOW")
        return "SLOW"


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
    ema_12 = calculate_ema(arr_prices[-12:], 12)  # Calculate the ema 12 with latest 12 candles
    ema_26 = calculate_ema(arr_prices, 26)  # Calculate the ema 26 with all 26 candles
    flo_macd_line = round((ema_12 - ema_26), 2)  # Calculate the macd-line
    lis_macd_lines.append(flo_macd_line)  # Add the macd-line to the list

    if len(lis_macd_lines) > 9:  # Keep only the last 9 values for the 9-period EMA calculation
        lis_macd_lines = lis_macd_lines[-9:]
        flo_signal_line = round(calculate_ema(lis_macd_lines, 9), 2)  # Calculate the signal line
        return(flo_macd_line, flo_signal_line)
    return (flo_macd_line, flo_signal_line)