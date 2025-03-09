# ========================================== Imports ===================================================================
import threading
from datetime import datetime, time
import time as sleep_time

import connection
import contracts
import market
import strategy
import statuscheck
import mainwindow
import ai_logging

# from ml_data.ai import start_ai


# ======================================== Global variables ============================================================


lst_fetched_candles = []                                                               # Global: has a fetched candles

# ========================================== Functions =================================================================

# The function 'startbutton_clicked' starts the thread with the window and the main loop
# def startbutton_clicked():
#     threading.Thread(target=start_loop, args=(session_manager,)).start()

def start_button_clicked():
    threading.Thread(target=start_loop).start()

def market_closed():
    print("Market closed")
    sleep_time.sleep(5400)  # Wait a one and half hour
    connection.create_new_session()  # Start a new session with the API (connection.py)
    account_information = connection.account_details()  # Get all the account information from the API (connection.py)
    start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
    return start_balance

# The function 'start loop' is the actual start of the bot.
def start_loop():
    boo_is_contract_open = False
    contract_id = ""
    macd = 0.0
    signal = 0.0
    advice = "HOLD"
    market_condition = "NEUTRAL"
    direction = "NEUTRAL"
    flo_contract_size = 0.0
    int_stop_loss_distance = 0
    int_this_contract_profit = 0
    int_positive_counter = 0
    str_contract_id = ""
    # print("Preparing AI")
    # start_ai()
    print("AI is NOT active and the bot has started making profit")                     # Marks the beginning of the loop
    connection.create_new_session()                                                     # Start a new session with the API (connection.py)
    account_information = connection.account_details()                                             # Get all the account information from the API (connection.py)
    start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)   # Get the current balance from the account
    flo_pre_contract_balance = account_information['accounts'][0]['balance']['balance']
    now = datetime.now()                                                                # Get the current date and time
    start_datetime = now.strftime("%d-%m-%Y %H:%M:%S")                                  # Format as DD-MM-YYYY HH:MM:SS
    int_number_of_candles = 0
    while True:  # Start of the rotation loop
        current_time = datetime.now().time()  # Get the current time
        if current_time >= time(22, 50) or current_time <= time(0, 10):  # Look at the current time
            start_balance = market_closed()
        else:
            lst_fetched_candles.insert(0, market.fetch_current_market_info())  # Get the candle information from the market (connection.py)
            int_number_of_candles += 1
        if len(lst_fetched_candles) > 25 and not boo_is_contract_open:  # If the candles counter reached higher then 25
            market_condition = market.detect_market_condition(lst_fetched_candles)
            advice, macd, signal = strategy.strategycheck(lst_fetched_candles)
            if market_condition == "NORMAL" and advice != "HOLD":
                direction, flo_contract_size, int_stop_loss_distance = contracts.open_new_contract(lst_fetched_candles, advice)
                boo_is_contract_open = True
                int_positive_counter = 0
                contract_id = ai_logging.log_contract_start(lst_fetched_candles, macd, signal, advice, current_time, flo_pre_contract_balance)
        if boo_is_contract_open:
            str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter = statuscheck.contractstatus(flo_pre_contract_balance, int_positive_counter)
            if new_profit_balance != flo_pre_contract_balance:
                boo_is_contract_open = False
                flo_pre_contract_balance = new_profit_balance
                ai_logging.log_contract_end(contract_id, str_contract_end)
        if len(lst_fetched_candles) > 30:  # If the array lst_fetched_candles is larger then 30
            lst_fetched_candles.pop()  # Trim lst_fetched_candles to contain only the last 30 candles
        if len(lst_fetched_candles) > 2:
            mainwindow.update_screen(lst_fetched_candles, macd, signal, advice, start_balance, start_datetime, int_number_of_candles, market_condition, direction, flo_contract_size, int_stop_loss_distance, int_this_contract_profit, int_positive_counter)
        sleep_time.sleep(180)  # Wait 5 minutes then start the loop again



# The function 'main' is the starting point of the program
def main():
    mainwindow.window.after(5000, start_button_clicked)
    mainwindow.window.mainloop()


# ========================================== Main start ================================================================

if __name__ == "__main__":
     main()