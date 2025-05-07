import csv
import os
from connection import create_new_session, fetch_us100_history, make_request
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime, timedelta
from sklearn.utils import resample


# Define the CSV file name
csv_filename = "../other/nasdaq_backtest.csv"


def calculate_atr(df, period=10):
    """Calculate ATR for a DataFrame with OHLC columns."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_ema(prices, period):
    weighting_factor = 2 / (period + 1)
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return round(ema[-1], 2)


bot_macd_lines = []

def calculate_macd_values(prices):
    global bot_macd_lines
    signal_line = 0.0

    if len(prices) < 26:
        return (0, 0)  # Not enough data

    ema_12 = calculate_ema(prices[:12], 12)
    ema_26 = calculate_ema(prices[:26], 26)
    macd_line = round(ema_12 - ema_26, 2)
    bot_macd_lines.append(macd_line)

    if len(bot_macd_lines) > 9:
        bot_macd_lines = bot_macd_lines[-9:]
        signal_line = round(calculate_ema(bot_macd_lines, 9), 2)

    return macd_line, signal_line

def calculate_macd_full_history(prices):
    """Compute MACD & Signal line over full price history."""
    if len(prices) < 26:
        return [], []  # Not enough data

    macd_lines = []
    signal_lines = []

    for i in range(26, len(prices)):
        ema_12 = calculate_ema(prices[i-12:i], 12)
        ema_26 = calculate_ema(prices[i-26:i], 26)
        macd_line = round(ema_12 - ema_26, 2)
        macd_lines.append(macd_line)

        if len(macd_lines) >= 9:
            signal_line = round(calculate_ema(macd_lines[-9:], 9), 2)
            signal_lines.append(signal_line)
        else:
            signal_lines.append(0.0)  # Until 9 MACD values exist

    return macd_lines, signal_lines


def macd_check(macd, signal):
    if macd > (signal + 1):
        return "BUY"
    elif macd < (signal - 1):
        return "SELL"
    else:
        return "HOLD"


def three_candle_movement(lst_fetched_candles):
    """Determine buy/sell signal based on the last 3 candles with an improved dynamic threshold."""

    atr = calculate_atr(lst_fetched_candles, 10)

    # Step 1: Compute the average open-close difference over the last 10 candles
    recent_candles = lst_fetched_candles[:10]
    avg_open_close_diff = sum(abs(candle[0] - candle[1]) for candle in recent_candles) / len(recent_candles)

    # ✅ Scale ATR to match actual price movements
    dynamic_threshold = max(atr * 0.0005, 2)  # 10% of ATR, with a minimum of 2
    # dynamic_threshold = max(2, min(avg_open_close_diff * 0.75, 6))

    # Step 3: Extract open prices for the last 3 candles
    current_candle_open = lst_fetched_candles[0][0]
    previous_candle_open = lst_fetched_candles[1][0]
    oldest_candle_open = lst_fetched_candles[2][0]

    # Step 4: Apply the adjusted dynamic threshold
    three_candle_advice = "HOLD"
    if current_candle_open > (previous_candle_open + 5) and previous_candle_open > (
            oldest_candle_open + 5):
        three_candle_advice = "BUY"
    elif current_candle_open < (previous_candle_open - 5) and previous_candle_open < (
            oldest_candle_open - 5):
        three_candle_advice = "SELL"

    return three_candle_advice


def strategycheck(lst_fetched_candles, macd, signal):
    macd_advice = macd_check(macd, signal)
    three_candle_advice = three_candle_movement(lst_fetched_candles)
    if macd_advice == three_candle_advice and macd_advice != "HOLD" and three_candle_advice != "HOLD":
        return macd_advice
    else:
        return "HOLD"


def calculate_macd_both_versions(prices):
    """Compute both full-history MACD and bot-style MACD for comparison."""

    if len(prices) < 26:
        return [], [], 0.0, 0.0  # Not enough data

    macd_lines = []
    signal_lines = []

    # 📌 Calculate full-history MACD
    for i in range(26, len(prices)):
        ema_12 = calculate_ema(prices[i - 12:i], 12)
        ema_26 = calculate_ema(prices[i - 26:i], 26)
        macd_line = round(ema_12 - ema_26, 2)
        macd_lines.append(macd_line)

        if len(macd_lines) >= 9:
            signal_line = round(calculate_ema(macd_lines[-9:], 9), 2)
            signal_lines.append(signal_line)
        else:
            signal_lines.append(0.0)  # Until 9 MACD values exist

    # 📌 Calculate bot-style MACD (last 26 candles only)
    bot_ema_12 = calculate_ema(prices[-12:], 12)
    bot_ema_26 = calculate_ema(prices[-26:], 26)
    bot_macd = round(bot_ema_12 - bot_ema_26, 2)
    bot_signal = round(calculate_ema(macd_lines[-9:], 9) if len(macd_lines) >= 9 else 0.0, 2)

    return macd_lines, signal_lines, bot_macd, bot_signal

def detect_market_condition(lis_fetched_candles):
    """Determine if the market is NORMAL or SLOW based on last 3 snapshots and log results."""

    from datetime import datetime  # Import inside to avoid unnecessary global dependency

    # 1. Calculate Volatility (Last 3 Snapshots)
    bids = [snap[1] for snap in lis_fetched_candles[:3]]
    volatility = (max(bids) - min(bids)) / min(bids) * 100

    # 2. Consecutive Percentage Change
    percentage_changes = [
        ((bids[i] - bids[i - 1]) / bids[i - 1]) * 100 for i in range(1, len(bids))
    ]
    fast_changes = sum(abs(change) >= 0.1 for change in percentage_changes)

    # Decision Logic
    if volatility > 0.05 and fast_changes >= 0.2:
        market_status = "NORMAL"
    else:
        market_status = "SLOW"
    return market_status

# List of known US stock market holidays in 2023/2024
MARKET_HOLIDAYS = {
    # 2023 Market Holidays
    "2023-01-02",  # New Year's Day (Observed)
    "2023-01-16",  # Martin Luther King Jr. Day
    "2023-02-20",  # Presidents' Day
    "2023-04-07",  # Good Friday
    "2023-05-29",  # Memorial Day
    "2023-06-19",  # Juneteenth
    "2023-07-04",  # Independence Day
    "2023-09-04",  # Labor Day
    "2023-11-23",  # Thanksgiving
    "2023-12-25",  # Christmas Day

    # 2024 Market Holidays
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents' Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas Day
}


def fetch_all_candles():
    """Fetch historical 5-minute candle data for each trading day in 2024 and store them."""
    create_new_session()  # Ensure API session is active

    start_date = datetime(2023, 1, 1)  # Start from January 1st, 2024
    end_date = datetime(2024, 12, 31)  # Until December 31st, 2024
    delta = timedelta(days=1)

    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")

        # Skip weekends (Saturday=5, Sunday=6) and holidays
        if start_date.weekday() >= 5 or date_str in MARKET_HOLIDAYS:
            print(f"Skipping {date_str} (Weekend or Holiday)")
        else:
            print(f"Fetching data for {date_str}...")

            # Fetch the daily candles
            candle_history = make_request(
                "GET",
                f"/api/v1/prices/US100?resolution=MINUTE_5&max=200&from={date_str}T10:00:00&to={date_str}T22:00:00"
            )

            if candle_history:
                print(f"✅ {len(candle_history)} candles fetched for {date_str}")
                create_history_csv(candle_history)  # Store in CSV
            else:
                print(f"⚠ No data for {date_str} (Possible unexpected closure)")

        start_date += delta  # Move to the next day

    print("✅ Full historical data fetch complete!")

def create_history_csv(candle_history):
        # Define column headers
    headers = ["Open", "Close", "High", "Low"]
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if file is empty
        if file.tell() == 0:
            writer.writerow(headers)

        # Extract and write relevant data
        for entry in candle_history["prices"]:
            writer.writerow([
                entry["openPrice"]["bid"],  # Use bid prices for consistency
                entry["closePrice"]["bid"],
                entry["highPrice"]["bid"],
                entry["lowPrice"]["bid"],
            ])

    print(f"✅ NASDAQ data saved to {csv_filename} for backtesting!")


def fetch_and_convert_to_3min_candles(candle_history):
    """Fetches 1-minute candle data, converts it to 3-minute candles, and saves it to a CSV file."""

    three_minute_candles = []
    one_minute_candles = []

    # ✅ Extract 1-minute candle data from API response
    for entry in candle_history["prices"]:
        one_minute_candles.append([
            entry["snapshotTimeUTC"],
            entry["openPrice"]["bid"],
            entry["closePrice"]["bid"],
            entry["highPrice"]["bid"],
            entry["lowPrice"]["bid"],
            entry["lastTradedVolume"],
        ])

    # ✅ Convert 1-minute candles into 3-minute candles
    for i in range(0, len(one_minute_candles) - 2, 3):
        open_price = one_minute_candles[i][1]  # First minute's open
        close_price = sum(one_minute_candles[i + j][2] for j in range(3)) / 3  # Average close
        high_price = max(one_minute_candles[i + j][3] for j in range(3))  # Highest high
        low_price = min(one_minute_candles[i + j][4] for j in range(3))  # Lowest low
        timestamp = one_minute_candles[i + 2][0]  # Timestamp of last candle in group
        volume = one_minute_candles[i + 2][5]

        three_minute_candles.append([timestamp, open_price, close_price, high_price, low_price, volume])

    # ✅ Save only the 3-minute candles to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Open", "Close", "High", "Low"])  # Header row
        writer.writerows(three_minute_candles)

def check_historical_closing():
    real_closing_prices = []
    estimated_closing_prices = []

    with open(csv_filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            open_price = float(row["Open"])
            close_price = float(row["Close"])  # Real closing price
            high_price = float(row["High"])
            low_price = float(row["Low"])

            # Simulate alternative closing price (avg of Open, High, Low)
            estimated_close = (open_price + high_price + low_price) / 3

            # Store values
            real_closing_prices.append(close_price)
            estimated_closing_prices.append(estimated_close)

    # Convert lists to NumPy arrays
    real_closing_prices = np.array(real_closing_prices)
    estimated_closing_prices = np.array(estimated_closing_prices)

    # Calculate accuracy metrics
    absolute_errors = np.abs(real_closing_prices - estimated_closing_prices)
    mae = np.mean(absolute_errors)  # Mean Absolute Error
    mse = np.mean((real_closing_prices - estimated_closing_prices) ** 2)  # Mean Squared Error
    accuracy = 100 - (np.mean(absolute_errors / real_closing_prices) * 100)  # Percentage Accuracy

    # Print results
    print(f"📊 Accuracy of Alternative Closing Price:")
    print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
    print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
    print(f"✅ Accuracy: {accuracy:.2f}%")

def check_historical_macd():
    real_closing_prices = []
    alt_closing_prices = []
    timestamps = []

    # Read data
    with open(csv_filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            timestamps.append(row["Time"])
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])

            # Real closing price
            real_closing_prices.append(close_price)

            # Alternative closing price (Open + High + Low) / 3
            alt_closing_prices.append((open_price + high_price + low_price) / 3)

    # Compute MACD for real and alternative closing prices (full history)
    macd_real, signal_real = calculate_macd_full_history(real_closing_prices)
    macd_alt, signal_alt = calculate_macd_full_history(alt_closing_prices)

    # Compare MACD values across history
    macd_errors = [abs(a - b) for a, b in zip(macd_real, macd_alt)]
    signal_errors = [abs(a - b) for a, b in zip(signal_real, signal_alt)]

    avg_macd_error = sum(macd_errors) / len(macd_errors)
    avg_signal_error = sum(signal_errors) / len(signal_errors)

    print(f"📊 MACD Comparison (Real vs. Alternative Closing Prices)")
    print(f"✅ Average MACD Error: {avg_macd_error:.2f}")
    print(f"✅ Average Signal Line Error: {avg_signal_error:.2f}")
    print(f"✅ Accuracy: {100 - (avg_macd_error / max(macd_real) * 100 if max(macd_real) != 0 else 0):.2f}%")


import statistics


def calculate_volatility(lst_fetched_candles, period=10):
    """
    Calculate volatility as the standard deviation of open-close price differences.
    """
    if len(lst_fetched_candles) < period:
        return 0  # Not enough data

    open_close_diffs = [abs(candle[0] - candle[1]) for candle in lst_fetched_candles[-period:]]

    return round(statistics.stdev(open_close_diffs), 2) if len(open_close_diffs) > 1 else 0



training_data_file = "../other/training_data.csv"


def backtest_strategy():
    """Backtest the strategy and save BUY, SELL, and HOLD cases for balanced AI training."""

    # Check if the file exists (to write headers only once)
    file_exists = os.path.isfile(training_data_file)

    # Load historical candles
    historical_candles = []
    with open(csv_filename, mode="r", newline="") as file:
        file.seek(0)
        reader = csv.DictReader(file)
        for row in reader:
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            historical_candles.append([open_price, close_price, high_price, low_price])

    # Track performance
    results = {
        "bot_macd": {
            "total": 0, "correct": 0, "rejected": 0,
            "rejected_due_to_market": 0, "profit": 0,
            "hold_count": 0  # Track HOLD scenarios
        }
    }

    # Open file for writing filtered trades
    with open(training_data_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers if the file is new
        if not file_exists:
            writer.writerow([
                "Prev2_Open", "Prev2_Close", "Prev2_High", "Prev2_Low",
                "Prev1_Open", "Prev1_Close", "Prev1_High", "Prev1_Low",
                "Current_Open", "Current_Close", "Current_High", "Current_Low",
                "ATR", "MACD", "Signal_Line", "Trade"
            ])

        # Backtest with at least 26 candles
        for i in range(26, len(historical_candles)):
            lst_fetched_candles = historical_candles[i - 26:i + 1]  # Use last 26 candles

            market_condition = detect_market_condition(lst_fetched_candles)
            if market_condition != "NORMAL":
                results["bot_macd"]["rejected_due_to_market"] += 1
                continue  # Skip this trade

            # Compute indicators
            bot_macd, bot_signal = calculate_macd_values(lst_fetched_candles)
            atr = calculate_atr(lst_fetched_candles, 14)
            volatility = calculate_volatility(lst_fetched_candles)

            # Run strategy
            strategy_signal_bot = strategycheck(lst_fetched_candles, bot_macd, bot_signal)

            # ✅ Save HOLD scenarios (even if no trade is taken)
            if strategy_signal_bot == "HOLD":
                results["bot_macd"]["hold_count"] += 1

                # Save HOLD data for training
                if i >= 2:  # Ensure at least 2 previous candles
                    previous_candle_2 = historical_candles[i - 2]
                    previous_candle_1 = historical_candles[i - 1]
                    current_candle = historical_candles[i]

                    writer.writerow([
                        previous_candle_2[0], previous_candle_2[1], previous_candle_2[2], previous_candle_2[3],
                        previous_candle_1[0], previous_candle_1[1], previous_candle_1[2], previous_candle_1[3],
                        current_candle[0], current_candle[1], current_candle[2], current_candle[3],
                        atr, bot_macd, bot_signal, "HOLD"  # Explicitly label as HOLD
                    ])
                continue  # Skip further processing for HOLD

            # Process BUY/SELL signals
            results["bot_macd"]["total"] += 1
            current_candle = historical_candles[i]

            # Simulate trade outcome (e.g., trailing stop)
            if i + 10 < len(historical_candles):
                next_10_candles = historical_candles[i + 1:i + 11]
            else:
                next_10_candles = historical_candles[i + 1:]

            exit_price, success, profit = simulate_trailing_stop_loss(
                lst_fetched_candles, current_candle[0], strategy_signal_bot, next_10_candles
            )

            results["bot_macd"]["profit"] += profit
            if success:
                results["bot_macd"]["correct"] += 1
            else:
                results["bot_macd"]["rejected"] += 1

            # Save BUY/SELL data for training
            if i >= 2:
                previous_candle_2 = historical_candles[i - 2]
                previous_candle_1 = historical_candles[i - 1]
                current_candle = historical_candles[i]

                writer.writerow([
                    previous_candle_2[0], previous_candle_2[1], previous_candle_2[2], previous_candle_2[3],
                    previous_candle_1[0], previous_candle_1[1], previous_candle_1[2], previous_candle_1[3],
                    current_candle[0], current_candle[1], current_candle[2], current_candle[3],
                    atr, bot_macd, bot_signal, strategy_signal_bot  # BUY or SELL
                ])

    # Print final results
    total = results["bot_macd"]["total"]
    correct = results["bot_macd"]["correct"]
    rejected = results["bot_macd"]["rejected"]
    market = results["bot_macd"]["rejected_due_to_market"]
    hold = results["bot_macd"]["hold_count"]
    profit = results["bot_macd"]["profit"]

    accuracy = (correct / total) * 100 if total > 0 else 0
    rejection_rate = (rejected / total) * 100 if total > 0 else 0

    print(f"\n📊 Backtest Results (Bot MACD)")
    print(f"✅ Total Trades Taken: {total}")
    print(f"✅ Correct Trades: {correct}")
    print(f"✅ Rejected Signals: {rejected}")
    print(f"🚫 Rejected Signals (Market Too Slow): {market}")
    print(f"⏸️  HOLD Signals: {hold}")
    print(f"🎯 Strategy Accuracy: {accuracy:.2f}%")
    print(f"🚫 Rejection Rate: {rejection_rate:.2f}%")
    print(f"\n💰 Total Profit/Loss: €{profit:.2f}")
    print(f"✅ Training data saved to {training_data_file}")


def calculate_trailing_stop_loss(lst_fetched_candles: list[list[float]], multiplier: int = 2, min_stop_loss: int = 25) -> float:
    """
    Calculate stop-loss distance based on recent open price differences.
    """
    recent_candles = lst_fetched_candles[:10]

    # Calculate absolute differences between consecutive open prices
    open_differences = [abs(recent_candles[i][0] - recent_candles[i + 1][0]) for i in range(len(recent_candles) - 1)]

    # Calculate the average difference
    avg_diff = sum(open_differences) / len(open_differences) if open_differences else 0

    # Scale the stop-loss and ensure it's above the minimum
    stop_loss_distance = round(avg_diff * multiplier, 0)
    int_stop_loss_distance = max(stop_loss_distance, min_stop_loss)

    # print(f"Avg Open Difference: {avg_diff:.2f}, Stop Loss Distance: {int_stop_loss_distance}")

    return int_stop_loss_distance


def simulate_trailing_stop_loss(lst_fetched_candles, entry_price, trade_type, candles, min_hold_time=10, contract_size=1):
    """
    Simulates a fixed trailing stop loss and calculates profit/loss.
    """
    if not candles:
        return entry_price, False, 0  # No future candles, assume trade failed

    # trailing_distance = calculate_trailing_stop_loss(lst_fetched_candles)
    trailing_distance = 10

    stop_loss = entry_price - trailing_distance if trade_type == "BUY" else entry_price + trailing_distance
    max_favorable_price = entry_price
    trade_duration = 0

    for i, candle in enumerate(candles):
        high, low = candle[2], candle[3]  # High and low prices
        trade_duration += 1

        if trade_type == "BUY":
            if high > max_favorable_price:
                max_favorable_price = high
                stop_loss = max(stop_loss, max_favorable_price - trailing_distance)
            if low <= stop_loss and trade_duration >= min_hold_time:
                profit = (stop_loss - entry_price) * contract_size  # ✅ P&L Calculation
                return stop_loss, stop_loss > entry_price, profit  # Exit at stop loss

        elif trade_type == "SELL":
            if low < max_favorable_price:
                max_favorable_price = low
                stop_loss = min(stop_loss, max_favorable_price + trailing_distance)
            if high >= stop_loss and trade_duration >= min_hold_time:
                profit = (entry_price - stop_loss) * contract_size  # ✅ P&L Calculation
                return stop_loss, stop_loss < entry_price, profit  # Exit at stop loss

    final_price = candles[-1][0]  # Exit at last available price
    profit = (final_price - entry_price) * contract_size if trade_type == "BUY" else (entry_price - final_price) * contract_size
    return final_price, final_price > entry_price, profit




def main():
    print("Testing a scenario 's")
    # candle_history = fetch_all_candles()
    # create_history_csv(candle_history)
    # fetch_and_convert_to_3min_candles(candle_history)
    # check_historical_closing()
    # check_historical_macd()
    # backtest_strategy()
    df = pd.read_csv("../other/nasdaq_backtest.csv")




if __name__ == "__main__":
     main()