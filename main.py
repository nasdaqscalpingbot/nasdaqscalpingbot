import sys
import threading
from datetime import datetime, time
import time as sleep_time

from PyQt5.QtWidgets import QApplication
from interface.scalpingbotview import ScalpingbotView  # Import the UI class

import connection
import contracts
import market
import strategy
import statuscheck
import ai_logging

# Global variable for fetched candles
lst_fetched_candles = []
boo_is_contract_open = False
contract_id = ""
macd = 0.1
signal = 0.1
advice = "HOLD"
market_condition = "NEUTRAL"
flo_contract_size = 0.0
flo_pre_contract_balance = 0.0
int_stop_loss_distance = 28
int_this_contract_profit = 0
int_positive_counter = 0
str_contract_id = ""

def get_contract_status():
    return boo_is_contract_open

def set_contract_status(state):
    global boo_is_contract_open
    boo_is_contract_open = state
    return



def start_loop(ui_window):
    """Main bot loop that runs in a separate thread and updates the UI."""
    global lst_fetched_candles
    global boo_is_contract_open
    global contract_id
    global macd
    global signal
    global advice
    global market_condition
    global flo_contract_size
    global flo_pre_contract_balance
    global int_stop_loss_distance
    global int_this_contract_profit
    global int_positive_counter
    global str_contract_id

    print("The bot with the AI has started.")
    connection.create_new_session()  # Start API session
    account_information = connection.account_details()
    start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
    flo_pre_contract_balance = start_balance
    now = datetime.now()
    start_datetime = now.strftime("%d-%m-%Y %H:%M:%S")
    int_number_of_candles = 0

    sleep_time.sleep(2)  # Allow UI to initialize before first update

    while True:
        current_time = datetime.now().time()

        # Market close handling
        if current_time >= time(22, 50) or current_time <= time(0, 5):
            print("Market closed")
            sleep_time.sleep(5400)  # Wait 1.5 hours
            connection.create_new_session()
            account_information = connection.account_details()
            start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
            flo_pre_contract_balance = start_balance
        else:
            #Fetch new market data
            lst_fetched_candles.insert(0, market.fetch_new_candle())
            int_number_of_candles += 1

        # Contract strategy
        if len(lst_fetched_candles) > 25:
            contract_status = get_contract_status()
            if not contract_status:
                # Open een nieuw contract
                market_condition = market.detect_market_condition(lst_fetched_candles)
                advice, macd, signal = strategy.strategycheck(lst_fetched_candles)
                if advice != "HOLD":
                    flo_contract_size, int_stop_loss_distance = contracts.open_new_contract(lst_fetched_candles, advice)
                    int_positive_counter = 0
                    # contract_id = ai_logging.log_contract_start(lst_fetched_candles, macd, signal, current_time, flo_pre_contract_balance)
                    set_contract_status(True)
            else:
                # Controleer de status van het contract
                str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter = statuscheck.contractstatus(flo_pre_contract_balance, int_positive_counter)
                if new_profit_balance != flo_pre_contract_balance:
                    set_contract_status(False)
                    flo_pre_contract_balance = new_profit_balance
                    # ai_logging.log_contract_end(contract_id, str_contract_end, int_this_contract_profit, new_profit_balance, lst_fetched_candles)
        # Limit candle list to last 30 entries
        if len(lst_fetched_candles) > 30:
            lst_fetched_candles.pop()

        """ mainwindow.update_screen(lst_fetched_candles, macd, signal, advice, start_balance, start_datetime,
                                 int_number_of_candles, market_condition, flo_contract_size,
                                 int_stop_loss_distance, int_this_contract_profit, int_positive_counter)"""
        #  Update the PyQt5 UI dynamically
        ui_window.update_ui_signal.emit(
            list(lst_fetched_candles[:]),
            float(macd),
            float(signal),
            str(advice),
            float(start_balance),
            str(start_datetime),
            int(int_number_of_candles),
            str(market_condition),
            float(flo_contract_size),
            int(int_stop_loss_distance),
            float(int_this_contract_profit),
            int(int_positive_counter)
        )


def main():
    """Start the PyQt5 UI and bot loop in separate threads."""
    app = QApplication(sys.argv)
    window = ScalpingbotView()
    # ✅ Start bot loop in a separate thread
    bot_thread = threading.Thread(target=start_loop, args=(window,))
    bot_thread.daemon = True
    bot_thread.start()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
