# ========================================== Imports ===================================================================
import statistics
import time
import json
import numpy as np
import datetime
from data.fetch_data import fetch_current_market_info
from connection import account_details, create_position, close_position, get_open_position, active_account
from data.csv_data import log_contract_start, log_contract_end
from interface.mainwindow import update_screen
# from ml_data.ai import make_a_prediction
from dataclasses import dataclass, field
# ======================================== Global variables ============================================================


@dataclass
class SessionData:
    lis_fetched_candles:        list[list] = field(default_factory=list)                                                # Contains all fetched candles(the last 30)
    lis_macd_lines:             list[list] = field(default_factory=list)                                                # Contains the macd-lines
    boo_is_contract_open:       bool = False                                                                            # Contains the status of a contract

    int_total_profit:           int = 0                                                                                 # Contains the total profit the bot has made since it started
    int_this_contract_profit:   int = 0                                                                                 # Contains the current profit of a ongoing contract
    int_positive_counter:       int = 0                                                                                 # Contains the number of times that a contract is in a positive state, 9 times max
    int_negative_counter:       int = 0                                                                                 # Contains the number of times that a contract is in a negative state, 9 times max
    int_contract_take_profit:   int = 0                                                                                 # Contains the value of a contract 'take profit'
    int_contract_stop_loss:     int = 0                                                                                 # Contains the value of a contract 'stop-loss'
    int_number_of_candles:      int = 0                                                                                 # Counter for the number of fetched candles
    int_after_contract_balance: int = 0                                                                                 # Contains the account balance after a contract is closed
    # int_take_profit_distance:   int = 0                                                                                 # Contains the distance of the contract 'take profit'
    # int_stop_loss_distance:     int = 0                                                                                 # Contains the distance of the contract 'stop loss'

    str_given_advice:           str = "HOLD"                                                                            # Holds the current contract advice (buy, sell, quickbuy, quicksell, hold)
    str_current_contract:       str = "HOLD"                                                                            # Holds the current contract advice (buy, sell, quickbuy, quicksell, hold)
    str_last_contract:          str = "HOLD"                                                                            # Holds the current contract advice (buy, sell, quickbuy, quicksell, hold)
    str_contract_id:            str = None                                                                              # Contains the string with the ID for the current open contract
    str_start_datetime:         str = ""

    flo_basic_contract_size:    float = 0.0                                                                             # Contains the size of the open contract
    flo_macd_line:              float = 0.0                                                                             # Contains the calculated macd-line
    flo_signal_line:            float = 0.0                                                                             # Contains the calculated signal line
    flo_histogram:              float = 0.0                                                                             # Contains the calculated histogram
    flo_pre_contract_balance:   float = 0.0                                                                             # Contains the account balance before the contract was opened
    flo_current_balance:        float = 0.0                                                                             # Contains the current account balance
    flo_start_balance:          float = 0.0                                                                             # Contains the account balance at the start of the bot

    current_candle_open:        float = 0.0
    current_candle_close:       float = 0.0
    current_candle_high:        float = 0.0
    current_candle_low:         float = 0.0
    previous_candle_open:       float = 0.0
    previous_candle_close:      float = 0.0
    previous_candle_high:       float = 0.0
    previous_candle_low:        float = 0.0
    oldest_candle_open:         float = 0.0
    oldest_candle_close:        float = 0.0
    oldest_candle_high:         float = 0.0
    oldest_candle_low:          float = 0.0
    current_candle_netChange:   float = 0.0
    previous_candle_netChange:  float = 0.0
    oldest_candle_netChange:    float = 0.0
    current_candle_PercentageChange:    float = 0.0
    avg_change:                 float = 0.0
    spread:                     float = 0.0
    std_dev:                    float = 0.0



S_SESSION: SessionData = SessionData()
candle_threshold:               int = 2
quick_candle_threshold:         int = 10
max_bid_ask_spread:             float = 0.05     # Maximum bid-ask spread allowed




# This function calculate the weighted ema
def calculate_ema(prices, period, weighting_factor=0.2):
    ema = np.zeros(len(prices))                                                                                         # Initialize an array of zeros to store the EMA values
    sma = np.mean(prices[:period])                                                                                      # Calculate the Simple Moving Average (SMA) for the first 'period' elements
    ema[period - 1] = sma                                                                                               # Set the initial EMA value at the end of the SMA period
    for i in range(period, len(prices)):                                                                                # Calculate the EMA for the remaining prices
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))                                 # Apply the EMA formula: (current price * weighting factor) + (previous EMA * (1 - weighting factor))
    return round(ema[-1], 1)                                                                                            # Return the last EMA value rounded to 1 decimal place


# This function calculates the MACD values
def calculate_macd_values():
    # Calculate the long-term EMA (26-period)
    arr_prices = [candle[1] for candle in S_SESSION.lis_fetched_candles[:26]]                                           # Get the latest 26 candles from the beginning of the list
    ema_12 = calculate_ema(arr_prices[-12:], 12)                                                                  # Calculate the ema 12 with latest 12 candles
    ema_26 = calculate_ema(arr_prices, 26)                                                                        # Calculate the ema 26 with all 26 candles
    S_SESSION.flo_macd_line = round((ema_12 - ema_26), 2)                                                               # Calculate the macd-line
    S_SESSION.lis_macd_lines.append(S_SESSION.flo_macd_line)                                                            # Add the macd-line to the list

    if len(S_SESSION.lis_macd_lines) > 9:                                                                               # Keep only the last 9 values for the 9-period EMA calculation
        S_SESSION.lis_macd_lines = S_SESSION.lis_macd_lines[-9:]
        S_SESSION.flo_signal_line = round(calculate_ema(S_SESSION.lis_macd_lines, 9), 2)                          # Calculate the signal line
        S_SESSION.flo_histogram = round(S_SESSION.flo_macd_line - S_SESSION.flo_signal_line, 2)                         # Calculate the histogram as the difference between MACD line and signal line
        return


def check_percentage_movement(flo_percentageChange):
    S_SESSION.avg_change = sum(candle[4] for candle in S_SESSION.lis_fetched_candles[:5]) / 5  # Calculate average
    print(f"Average Change: {S_SESSION.avg_change}, Current Change: {flo_percentageChange}")

    # Static threshold (if you want to test raising it)
    static_threshold = 50.0  # Adjust based on observed data

    # Dynamic threshold (optional)
    dynamic_threshold = S_SESSION.avg_change * 0.1  # 10% of avg_change

    # Use the dynamic threshold (or the static one)
    threshold = max(static_threshold, dynamic_threshold)

    # Check percentage movement
    if S_SESSION.avg_change < -25.0:
        return True
    else:
        return False


def check_bid_ask_spread(bid, ask):
    S_SESSION.spread = ask - bid
    print(f"Spread: {S_SESSION.spread}")

    # Dynamically adjust the spread threshold
    adaptive_threshold = max(max_bid_ask_spread, bid * 0.001)  # 0.1% of bid OR fixed max

    # Return True if the spread is SMALLER than or equal to the threshold
    if S_SESSION.spread <= adaptive_threshold:
        return True  # Active market
    else:
        return False  # Inactive market

def is_market_sideways():
    price_changes = [candle[1] - candle[0] for candle in S_SESSION.lis_fetched_candles[:10]]  # Close - Open for last 10 candles
    S_SESSION.std_dev = statistics.stdev(price_changes)
    print(f"Standard Deviation: {S_SESSION.std_dev}")
    if S_SESSION.std_dev == 0.0:
        return True  # No movement at all
    elif 0.0 < S_SESSION.std_dev < 0.05:
        return True  # Minimal movement
    elif 0.05 <= S_SESSION.std_dev < 0.1:
        return False
    else:
        return False

def buy_candle_check(current_candle, previous_candle, oldest_candle):
    if (current_candle - previous_candle) > candle_threshold and (previous_candle - oldest_candle) > candle_threshold and S_SESSION.current_candle_PercentageChange > 1:
        return True
    else:
        return False

def sell_candle_check(current_candle, previous_candle, oldest_candle):
    if (previous_candle - current_candle) > candle_threshold and (oldest_candle - previous_candle) > candle_threshold and S_SESSION.current_candle_PercentageChange < -1:
        return True
    else:
        return False


def buy_macd_evaluation():
    if S_SESSION.flo_macd_line != 0.0 and S_SESSION.flo_signal_line != 0.0:
        if S_SESSION.flo_macd_line > S_SESSION.flo_signal_line and abs(S_SESSION.flo_macd_line - S_SESSION.flo_signal_line) > 3:
            return True
    return False


def sell_macd_evaluation():
    if S_SESSION.flo_macd_line != 0.0 and S_SESSION.flo_macd_line != 0.0:
        if S_SESSION.flo_macd_line < S_SESSION.flo_signal_line and S_SESSION.flo_signal_line < 1:
            return True
    return False

def quick_buy_check(current_candle, previous_candle):
    if (current_candle - previous_candle) > quick_candle_threshold and S_SESSION.flo_signal_line > 5:
        return True
    else:
        return False

def quick_sell_check(current_candle, previous_candle):
    if (previous_candle - current_candle) > quick_candle_threshold and S_SESSION.flo_signal_line < -5:
        return True
    else:
        return False

# This function applies the conditions for the start of a buy or sell of the contract
def contract_strategy():
    if S_SESSION.int_number_of_candles > 26:                                                                          # Wait until the number of candles are at least 26
        calculate_macd_values()                                                                                         # Calculate the MACD values (def:calculate_macd_values)
        S_SESSION.str_given_advice = "HOLD"

        # Begin logic evaluation
        boo_buy_candle = False
        boo_sell_candle = False
        boo_quick_buy = False
        boo_quick_sell = False
        boo_buy_macd = False
        boo_sell_macd = False
        boo_valid_movement = False
        boo_valid_spread = False
        boo_market_sideways = True

        current_candle = S_SESSION.current_candle_open
        current_close = S_SESSION.current_candle_close
        previous_candle = S_SESSION.previous_candle_open
        oldest_candle = S_SESSION.oldest_candle_open
        flo_percentageChange = S_SESSION.current_candle_PercentageChange

        boo_valid_movement = check_percentage_movement(flo_percentageChange)
        boo_valid_spread = check_bid_ask_spread(current_candle, current_close)
        boo_market_sideways = is_market_sideways()

        if not boo_valid_movement:
            print("Skipped trade due to insufficient movement.")
        # if not boo_valid_spread:
        #     print("Skipped trade due to too wide spread.")
        # if boo_market_sideways:
        #     print("Skipped trade due to sideways market.")
        if boo_valid_movement: #and boo_valid_spread and not boo_market_sideways:
            boo_buy_candle = buy_candle_check(current_candle, previous_candle, oldest_candle)
            boo_sell_candle = sell_candle_check(current_candle, previous_candle, oldest_candle)
            boo_quick_buy = quick_buy_check(current_candle, previous_candle)
            boo_quick_sell = quick_sell_check(current_candle, previous_candle)
            boo_buy_macd = buy_macd_evaluation()
            boo_sell_macd = sell_macd_evaluation()

        if (boo_buy_candle and boo_buy_macd) or boo_quick_buy:
            S_SESSION.str_given_advice = "BUY"
        elif (boo_sell_candle and boo_sell_macd) or boo_quick_sell:
            S_SESSION.str_given_advice = "SELL"


    # Verify the advice at the AI
    # if (S_SESSION.str_given_advice == "BUY" or S_SESSION.str_given_advice == "SELL"):
    #     prediction = make_a_prediction(
    #         [[S_SESSION.current_candle_open,
    #         S_SESSION.current_candle_close,
    #         S_SESSION.current_candle_high,
    #         S_SESSION.current_candle_low,
    #         S_SESSION.previous_candle_open,
    #         S_SESSION.previous_candle_close,
    #         S_SESSION.previous_candle_high,
    #         S_SESSION.previous_candle_low,
    #         S_SESSION.oldest_candle_open,
    #         S_SESSION.oldest_candle_close,
    #         S_SESSION.oldest_candle_high,
    #         S_SESSION.oldest_candle_low,
    #         S_SESSION.current_candle_netChange,
    #         S_SESSION.previous_candle_netChange,
    #         S_SESSION.oldest_candle_netChange,
    #         S_SESSION.flo_macd_line,
    #         S_SESSION.flo_signal_line,
    #         S_SESSION.flo_histogram]],
    #         S_SESSION.str_given_advice
    #     )
    #     if prediction in ["BUY", "SELL", "HOLD"]:
    #         S_SESSION.str_given_advice = prediction
    #     else:
    #         print(f"Unexpected prediction: {prediction}. Defaulting to HOLD.")
    #         S_SESSION.str_given_advice = "HOLD"
    return


# This function handles the final steps after a contract is closed
def end_contract():
    S_SESSION.int_positive_counter = 0                                                                                  # Reset the counter to for positive profit after 10 minutes
    S_SESSION.int_negative_counter = 0                                                                                  # Reset the counter for a negative or too much loss
    S_SESSION.boo_is_contract_open = False                                                                              # Change the state of the contract boolean
    log_contract_end(S_SESSION.str_contract_id, S_SESSION.int_this_contract_profit, S_SESSION.str_given_advice)         # Log the negative contract
    S_SESSION.str_given_advice = "Neutral"
    S_SESSION.int_this_contract_profit = 0
    return


# This function check the current profit/loss state of the open contract and take action if necessary
def statuscheck():
    account_information = account_details()                                                                             # Get the current account information
    S_SESSION.int_this_contract_profit = account_information['accounts'][0]['balance']['profitLoss']                    # Get the current profit/loss value of the open contract
    if S_SESSION.int_this_contract_profit == 0.0:                                                                       # If the value = 0, either thru stop-loss or take profit
        calculate_new_profit_balance()                                                                                  # Recalculate this contract profit
        if S_SESSION.int_this_contract_profit < 0:                                                                      # The profit was actual negative
            S_SESSION.str_given_advice = "Stop-loss"                                                                    # The contract ended because of the stop-loss
        else:                                                                                                           # else
            S_SESSION.str_given_advice = "Take profit"                                                                  # The contract ended because of the take profit
        end_contract()
        return
    elif S_SESSION.int_this_contract_profit > 0:                                                                        # If the current contract is positive
        S_SESSION.boo_is_contract_open = True                                                                           # Change the status of the contract
        S_SESSION.int_positive_counter += 1                                                                             # Increment of the positive counter
        S_SESSION.int_negative_counter = 0                                                                              # Reset the negative counter
        if S_SESSION.int_positive_counter >= 9:                                                                         # If the positive counter reach 9 (10 minutes)
            S_SESSION.str_contract_id = get_open_position()                                                             # Get the contract ID for the current contract
            close_position(S_SESSION.str_contract_id)                                                                   # Close the current contract
            calculate_new_profit_balance()                                                                              # Calculate the new contract value
            S_SESSION.str_given_advice = "Take profit 10"                                                               # The contract ended because of 10 minutes profit
            end_contract()
        return
    elif S_SESSION.int_this_contract_profit < 0:                                                                        # If the current contract is negative
        S_SESSION.boo_is_contract_open = True                                                                           # Change the status of the contract
        S_SESSION.int_negative_counter += 1                                                                             # Increment of the negative counter
        S_SESSION.int_positive_counter = 0                                                                              # Reset the positive counter
        if S_SESSION.int_negative_counter >= 9:                                                                         # If the negative counter reach 9 (10 minutes)
            S_SESSION.str_contract_id = get_open_position()                                                             # Get the contract ID for the current contract
            close_position(S_SESSION.str_contract_id)                                                                   # Close the current contract
            calculate_new_profit_balance()                                                                              # Calculate the new contract value
            S_SESSION.str_given_advice = "Stop loss 10"                                                                 # The contract ended because of the 10 minutes of loss
            end_contract()
        return
    return


# This function calculates the value of the profit/loss from the last closed contract
def calculate_new_profit_balance():
    account_information = account_details()                                                                             # Get the account information
    S_SESSION.int_after_contract_balance = round(account_information['accounts'][0]['balance']['balance'], 2)           # Get the current balance
    S_SESSION.int_this_contract_profit = round((S_SESSION.int_after_contract_balance - S_SESSION.flo_pre_contract_balance), 2)  # Calculate the profit or loss
    return


# This function calculate the contract details
def calculate_position_sizes():
    account_information = account_details()                                                                             # Get the current account details/balance
    S_SESSION.flo_pre_contract_balance = account_information['accounts'][0]['balance']['balance']                       # Store the current account balance
    S_SESSION.flo_basic_contract_size = round((S_SESSION.flo_pre_contract_balance - 300) / 1000, 1)                     # Contract size
    S_SESSION.int_contract_take_profit = round(S_SESSION.flo_pre_contract_balance * 0.02, 0)                            # Take profit size, 2% of the saldo
    S_SESSION.int_contract_stop_loss = round(S_SESSION.flo_pre_contract_balance * 0.01, 0)                              # Stop loss size, 1% of the saldo
    return


# This function handle the sequence of setting up a new contract
def open_new_contract():
    calculate_position_sizes()                                                                                          # Starts with calculating the contract values
    active_account()                                                                                                    # Activate the active account
    payload = json.dumps({                                                                                              # Prepare the position request
        "epic": "US100",                                                                                                # Set the market
        "direction": S_SESSION.str_given_advice,                                                                        # Set the direction buy or sell
        "size": S_SESSION.flo_basic_contract_size,                                                                      # Set the contracts size
        "level": 20,                                                                                                    # Set the level to 20, required, don't know what is does
        "type": "LIMIT",                                                                                                # Make it a limit order
        "stopAmount": S_SESSION.int_contract_stop_loss,                                                                 # Set the stop-loss amount
        "profitAmount": S_SESSION.int_contract_take_profit                                                              # Set the take profit amount
    })
    create_position(payload)                                                                                            # Create the actual contract (connection.py)
    S_SESSION.str_contract_id = get_open_position()                                                                     # Store the contract ID (connection.py)
    S_SESSION.boo_is_contract_open = True  # Change the open contract status
    S_SESSION.str_current_contract = S_SESSION.str_given_advice                                                     # Update screen information
    log_contract_start(S_SESSION)                                                                                   # Log the contract conditions
    return


# This function updates all the required values for the AI logging
def update_candle_values():
    # The content of one candle = [0]int_bid [1]int_close, [2]int_high, [3]int_low, [4]int_netChange, [5]int_percentageChange, [6]time]
    S_SESSION.current_candle_time = S_SESSION.lis_fetched_candles[0][6]
    S_SESSION.current_candle_open = S_SESSION.lis_fetched_candles[0][0]
    S_SESSION.current_candle_close = S_SESSION.lis_fetched_candles[0][1]
    S_SESSION.current_candle_high = S_SESSION.lis_fetched_candles[0][2]
    S_SESSION.current_candle_low = S_SESSION.lis_fetched_candles[0][3]
    S_SESSION.current_candle_netChange = S_SESSION.lis_fetched_candles[0][4]
    S_SESSION.current_candle_PercentageChange = S_SESSION.lis_fetched_candles[0][5]
    S_SESSION.previous_candle_open = S_SESSION.lis_fetched_candles[1][0]
    S_SESSION.previous_candle_close = S_SESSION.lis_fetched_candles[1][1]
    S_SESSION.previous_candle_high = S_SESSION.lis_fetched_candles[1][2]
    S_SESSION.previous_candle_low = S_SESSION.lis_fetched_candles[1][3]
    S_SESSION.previous_candle_netChange = S_SESSION.lis_fetched_candles[1][4]
    S_SESSION.oldest_candle_open = S_SESSION.lis_fetched_candles[2][0]
    S_SESSION.oldest_candle_close = S_SESSION.lis_fetched_candles[2][1]
    S_SESSION.oldest_candle_high = S_SESSION.lis_fetched_candles[2][2]
    S_SESSION.oldest_candle_low = S_SESSION.lis_fetched_candles[2][3]
    S_SESSION.oldest_candle_netChange = S_SESSION.lis_fetched_candles[2][4]
    return


# This function handles the sequence when there is a contract open
def handle_existing_contract():
    while S_SESSION.boo_is_contract_open:                                                                               # There is a contract open
        S_SESSION.lis_fetched_candles.insert(0, fetch_current_market_info())                                     # keep fetching candles for ema/macd
        update_candle_values()                                                                                          # Update the candle session values
        S_SESSION.int_number_of_candles += 1                                                                            # Increase the candle counter
        if S_SESSION.int_number_of_candles > 26:                                                                        # Wait until 26 candles
            calculate_macd_values()                                                                                     # Before calculating the MACD
        statuscheck()                                                                                                   # Do a check of the current profit/loss values
        if not S_SESSION.boo_is_contract_open:
            contract_strategy()
            if S_SESSION.str_given_advice == "BUY" or S_SESSION.str_given_advice == "SELL":  # If there is a buy or sell advice
                open_new_contract()  # Open a new contract
                handle_existing_contract()
            return
        else:
            S_SESSION.str_given_advice = "HOLD"                                                                         # Status is on hold
            log_contract_start(S_SESSION)  # Log the (new) current candle values
            update_screen(S_SESSION)                                                                                    # Update the screen
            time.sleep(60)                                                                                              # Wait 60 seconds and then loop again
    return


# This function is setting up the actual contract with the brooker
def setup_a_contract(fetched_candles, start_balance, start_time):
    current_time = datetime.datetime.now().time()                                                                       # Get the current time
    global S_SESSION                                                                                                    # Make the SESSION a global
    S_SESSION.str_start_datetime = start_time
    S_SESSION.lis_fetched_candles = fetched_candles                                                                     # Get a new candle
    if not S_SESSION.boo_is_contract_open and S_SESSION.int_number_of_candles > 2:
        statuscheck()
    update_candle_values()                                                                                              # Update the candle values
    S_SESSION.flo_start_balance = start_balance                                                                         # Store the start balance
    S_SESSION.int_number_of_candles += 1                                                                                # Increase the candle counter
    if current_time >= datetime.time(22, 45) or current_time <= datetime.time(00, 10):          # Look at the current time
        if S_SESSION.boo_is_contract_open and S_SESSION.str_current_contract == "BUY":                                  # If there is a contract still open, close the contract before the overnight costs
            close_position(S_SESSION.str_contract_id)                                                                   # Close the position
            calculate_new_profit_balance()                                                                              # Calculate the new contract value
            S_SESSION.str_given_advice = "Closed overnight"                                                             # The contract ended because of 10 minutes profit
            end_contract()                                                                                              # Log and reset
    if not S_SESSION.boo_is_contract_open:                                                                          # If there is not an open contract
        contract_strategy()                                                                                         # Before a buy or sell, give advice
        if S_SESSION.str_given_advice == "BUY" or S_SESSION.str_given_advice == "SELL":                             # If there is a buy or sell advice
           open_new_contract()                                                                                     # Open a new contract
           handle_existing_contract()                                                                                  # There is a contract open
    else:
        handle_existing_contract()
    update_screen(S_SESSION)                                                                                            # Update the screen
    return

"""
1. EMA Logic:
In both BUY and SELL conditions, the EMA logic compares S_SESSION.ema9 to S_SESSION.current_candle_open with very similar logic.
However, the SELL logic expects S_SESSION.ema9 to be greater than S_SESSION.current_candle_open, 
while the BUY logic expects S_SESSION.ema9 to be less than the candle open price.
The thresholds (ema_threshold) might be causing both conditions to evaluate similarly, 
or the values themselves might be too close to trigger a clear distinction between a BUY and SELL.
2. MACD Logic:
Similar to the EMA, the MACD conditions in both BUY and SELL compare S_SESSION.macd["flo_macd_line and S_SESSION.macd["flo_signal_line,
 but their logic seems valid. It expects:
BUY: flo_macd_line > flo_signal_line
SELL: flo_macd_line < flo_signal_line
Ensure that the thresholds (macd_threshold) are not too small, causing the difference between flo_macd_line and flo_signal_line to be negligible.

In general:
MACD line above signal line → Considered bullish(buy).
and S_SESSION.macd["macd > S_SESSION.macd["flo_signal_line):
MACD line below signal line → Considered bearish(sell).
and S_SESSION.macd["macd < S_SESSION.macd["flo_signal_line):

        if not S_SESSION.boo_is_contract_open:                                                                          # The status check closed a contract
            # calculate_new_profit_balance()                                                                              # Calculate
            # S_SESSION.str_given_advice = "CLOSED"
            # if S_SESSION['int_this_contract_profit'] > 0 and S_SESSION['int_after_contract_balance'] > 0:
            #     S_SESSION.int_total_profit = round(
            #         (S_SESSION.int_total_profit + S_SESSION['int_this_contract_profit']), 2)
            #     log_contract_end(S_SESSION.str_contract_id, S_SESSION['int_this_contract_profit'], "Profit")
            # elif S_SESSION['int_this_contract_profit'] < 0:
            #     S_SESSION.int_total_profit = round(
            #         (S_SESSION.int_total_profit + S_SESSION['int_this_contract_profit']), 2)
            #     log_contract_end(S_SESSION.str_contract_id, S_SESSION['int_this_contract_profit'], "Loss")
            #     S_SESSION['int_after_contract_balance'] = 0
            update_screen(S_SESSION)
        else:

    
    Buy strategy:
    If the current candle larger then the previous candle + 2
    and the previous candle is larger then the oldest candle + 2
    and the MACD line larger then the signal line
    and the MACD histogram larger then 3
    or
    if the current candle is 10 points larger the the previous candle
    Sell strategy:
    If the current candle smaller then the previous candle - 2
    and the previous candle is smaler then the oldest candle - 2
    and the MACD line smaller then the signal line
    and the signal line smaller then 1
    or:
    if the current candle is 10 point lower then the previous candle
    """