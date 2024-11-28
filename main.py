# ========================================== Imports ===================================================================
import threading
from datetime import datetime, time
import time as sleep_time
from connection import create_new_session, account_details
from data.fetch_data import fetch_current_market_info
from interface.mainwindow import window
from contracts.contracts import setup_a_contract
# from ml_data.ai import start_ai


# ======================================== Global variables ============================================================


lst_fetched_candles = []                                                               # Global: has a fetched candles


# ========================================== Functions =================================================================

# The function 'startbutton_clicked' starts the thread with the window and the main loop
def startbutton_clicked():
    threading.Thread(target=start_loop, args=()).start()


# The function 'delayed start' is activated if the start button is not clicked from within the program (which is never))
def delayed_start():
    startbutton_clicked()


# The function 'start loop' is the actual start of the bot.
def start_loop():
    global lst_fetched_candles                                                          # A list with all candles from the API
    # print("Preparing AI")
    # start_ai()
    print("AI is active and the bot has started making profit")                                          # Marks the beginning of the loop
    create_new_session()                                                                # Start a new session with the API (connection.py)
    account_information = account_details()                                             # Get all the account information from the API (connection.py)
    start_balance = round(account_information['accounts'][0]['balance']['balance'],2)   # Get the current balance from the account
    now = datetime.now()                                                                # Get the current date and time
    start_datetime = now.strftime("%d-%m-%Y %H:%M:%S")                                  # Format as DD-MM-YYYY HH:MM:SS
    while True:                                                                         # Start of the rotation loop
        current_time = datetime.now().time()  # Get the current time
        if current_time >= time(22, 50) or current_time <= time(0, 10):  # Look at the current time
            print("Market closed")
            sleep_time.sleep(5400)                                                      # Wait a one and half hour
            create_new_session()                                                        # Start a new session with the API (connection.py)
            account_details()                                                           # Get all the account information from the API (connection.py)
        lst_fetched_candles.insert(0, fetch_current_market_info())               # Get the candle information from the market (connection.py)
        int_candle_counter = len(lst_fetched_candles)                                   # Count the current number of candles
        if int_candle_counter > 2:                                                      # If the candles counter reached higher then 2
            setup_a_contract(lst_fetched_candles, start_balance, start_datetime)        # Set up a new contract (contracts.py)
        if int_candle_counter > 30:                                                     # If the array lst_fetched_candles is larger then 30
            lst_fetched_candles.pop()                                                  # Trim lst_fetched_candles to contain only the last 30 candles
        sleep_time.sleep(180)                                                                 # Wait 3 minutes then start the loop again

# The function 'main' is the starting point of the program
def main():
    window.after(5000, delayed_start)                                                   # Automatically call startbutton_clicked() after 5 seconds if not clicked manually
    window.mainloop()                                                                   # Start the main loop


# ========================================== Main start ================================================================

if __name__ == "__main__":
     main()