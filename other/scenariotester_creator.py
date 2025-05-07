import pandas as pd
import csv
import numpy as np
import os



def fetch_historical_data():
   pass


def write_historical_csv(historical_proces):
  pass


def calculate_ema(prices, period, weighting_factor=0.2):
    ema = np.zeros(len(prices))
    sma = np.mean(prices[:period])
    ema[period - 1] = sma
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))
    return round(ema[-1], 1)  #  round to 1 decimal


def calculate_macd(candles):
    # Get closing prices for EMAs
    ema_prices_9 = candles[-10:]
    ema_prices_10 = candles[-10:]

    # Calculate short-term EMA
    ema_9 = calculate_ema(ema_prices_9, 9)
    ema_10 = calculate_ema(ema_prices_10, 10)

    # Calculate long-term EMA
    arr_ema = candles[-20:]
    ema_20 = calculate_ema(arr_ema, 20)

    # MACD line (difference between the short and long EMA)
    macd_line = round((ema_10 - ema_20),1)

    # MACD histogram (difference between MACD line and signal line(EMA9]))
    macd_histogram = round((macd_line - ema_9),1)

    return {
        'ema9': ema_9,
        'MACD-line': macd_line,
        'signal-line': ema_9,
        'MACD-histogram': macd_histogram
    }


def write_csv_row_macd(update_row,macd_data):
    file_exists = os.path.isfile('scenariotest_data.csv')
    # Then write the calculated MACD to a new CSV file
    with open('scenariotest_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['snapshotTime', 'openPrice', 'closePrice',  'highPrice', 'lowPrice', 'ema9', 'MACD-line', 'signal-line', 'MACD-histogram'])

        writer.writerow([
            update_row['snapshotTime'],
            update_row['openPrice'],
            update_row['closePrice'],
            update_row['highPrice'],
            update_row['lowPrice'],
            macd_data['ema9'],
            macd_data['MACD-line'],
            macd_data['signal-line'],
            macd_data['MACD-histogram']
        ])


def add_ema9_macd():
    seeknumber = 20
    row_index = 0
    all_close_candles = []

    with open('us100_history.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row_index)
            if row_index > seeknumber:  # Skip the first 20 rows
                 all_close_candles.append(int(float(row['closePrice'])))
                 next20candles = all_close_candles[-20:]
                 macd_data = calculate_macd(next20candles)
                 update_row = row
                 write_csv_row_macd(update_row, macd_data)
            elif row_index == seeknumber:
                 macd_data = calculate_macd(all_close_candles)
                 write_csv_row_macd(update_row, macd_data)
            else:
                 all_close_candles.append(int(float(row['closePrice'])))
                 update_row = row
            row_index += 1



#
# # Function to process candle data
# def process_candles(arr_fetched_candles):
#     capital = 1000
#     trades = []
#
#     for i in range(20, len(arr_fetched_candles)):
#         # Extract the last 20 candles for EMA/MACD calculations
#         last_20_candles = arr_fetched_candles[i - 20:i]
#         close_prices = [candle[1] for candle in last_20_candles]
#
#         # Calculate EMA9 and MACD
#         ema9 = calculate_ema(close_prices, 9).iloc[-1]
#         macd_line, signal_line, macd_histogram = calculate_macd(close_prices)
#
#         # Apply your logic for buy/sell decisions
#         # You can add your custom buy/sell logic here
#         # Add your decision logic in this block:
#         # ------------------------------------------------------------------
#         # Example:
#         # if <BUY LOGIC BASED ON EMA9 and MACD>:
#         #     action = "BUY"
#         # elif <SELL LOGIC BASED ON EMA9 and MACD>:
#         #     action = "SELL"
#         # else:
#         #     action = "HOLD"
#         # ------------------------------------------------------------------
#         action = None  # Placeholder; replace with your logic
#
#         if action in ["BUY", "SELL"]:
#             # Calculate contract size, take profit, and stop-loss
#             # Add your logic for contract settings here:
#             # ------------------------------------------------------------------
#             # contract_size = <YOUR_LOGIC_FOR_CONTRACT_SIZE>
#             # take_profit = <YOUR_LOGIC_FOR_TAKE_PROFIT>
#             # stop_loss = <YOUR_LOGIC_FOR_STOP_LOSS>
#             # ------------------------------------------------------------------
#             contract_size = 0  # Placeholder; replace with your logic
#             take_profit = 0  # Placeholder; replace with your logic
#             stop_loss = 0  # Placeholder; replace with your logic
#
#             # Simulate the trade with these settings
#             # ------------------------------------------------------------------
#             # Run through subsequent candles to see if take profit or stop-loss is hit
#             # Update capital based on the outcome of the trade
#             # ------------------------------------------------------------------
#             trade_result = "profit"  # Or "loss" based on the simulation
#             profit_or_loss = 0  # Set the actual profit/loss from the trade
#
#             # Save trade data
#             trade_data = {
#                 "candle_timestamp": arr_fetched_candles[i][0],  # Assuming the first entry is the timestamp
#                 "ema9": ema9,
#                 "macd_line": macd_line.iloc[-1],
#                 "signal_line": signal_line.iloc[-1],
#                 "action": action,
#                 "contract_size": contract_size,
#                 "take_profit": take_profit,
#                 "stop_loss": stop_loss,
#                 "result": trade_result,
#                 "profit_or_loss": profit_or_loss
#             }
#             trades.append(trade_data)
#
#     return trades
#
#
# # Function to save the result to CSV
# def save_to_csv(trades, filename="trades_simulation.csv"):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=[
#             "candle_timestamp", "ema9", "macd_line", "signal_line", "action",
#             "contract_size", "take_profit", "stop_loss", "result", "profit_or_loss"
#         ])
#         writer.writeheader()
#         for trade in trades:
#             writer.writerow(trade)



def main():
    print("Scenariotester")
    # create_new_session()
    # dic_fetched_history = fetch_us100_history()
    dic_fetched_history = fetch_historical_data()
    write_historical_csv(dic_fetched_history)
    add_ema9_macd()



    # Sample candles (replace with your real API fetch logic)
    # Format: [timestamp, close, high, low]
    arr_fetched_candles = [
        # Add actual candle data here fetched from the API
    ]

    # # Process the fetched candles and run the simulation
    # trades = process_candles(arr_fetched_candles)
    #
    # # Save the simulated trades to CSV
    # save_to_csv(trades)


if __name__ == "__main__":
     main()
