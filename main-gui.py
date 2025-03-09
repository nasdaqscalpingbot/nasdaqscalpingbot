import sys
import threading
from datetime import datetime, time
import time as sleep_time

from PyQt5.QtWidgets import QApplication
from interface.copy_scalpingbotview import ScalpingbotView  # Import the UI class

import connection
import contracts
import market
import strategy
import statuscheck
import ai_logging

# Global variable for fetched candles
lst_fetched_candles = []


def start_loop(ui_window):
    """Main bot loop that runs in a separate thread and updates the UI."""
    boo_is_contract_open = False
    contract_id = ""
    macd = 0.1
    signal = 0.1
    advice = "HOLD"
    market_condition = "NEUTRAL"
    direction = "NEUTRAL"
    flo_contract_size = 0.0
    int_stop_loss_distance = 0
    int_this_contract_profit = 0
    int_positive_counter = 0
    str_contract_id = ""

    print("AI is NOT active, but the bot has started making profit.")
    connection.create_new_session()  # Start API session
    account_information = connection.account_details()
    start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
    flo_pre_contract_balance = start_balance
    now = datetime.now()
    start_datetime = now.strftime("%H:%M:%S")
    int_number_of_candles = 0

    sleep_time.sleep(2)  # Allow UI to initialize before first update

    while True:
        current_time = datetime.now().time()

        # Market close handling
        if current_time >= time(22, 50) or current_time <= time(0, 10):
            print("Market closed")
            sleep_time.sleep(5400)  # Wait 1.5 hours
            connection.create_new_session()
            account_information = connection.account_details()
            start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
            flo_pre_contract_balance = start_balance
        else:
            # Fetch new market data
            lst_fetched_candles.insert(0, market.fetch_current_market_info())
            # lst_fetched_candles.insert(0, market.fetch_new_candle())
            int_number_of_candles += 1

        # Contract strategy
        if len(lst_fetched_candles) > 25 and not boo_is_contract_open:
            market_condition = market.detect_market_condition(lst_fetched_candles)
            advice, macd, signal = strategy.strategycheck(lst_fetched_candles)

            if market_condition == "NORMAL" and advice != "HOLD":
                direction, flo_contract_size, int_stop_loss_distance = contracts.open_new_contract(lst_fetched_candles,
                                                                                                   advice)
                boo_is_contract_open = True
                int_positive_counter = 0
                contract_id = ai_logging.log_contract_start(lst_fetched_candles, macd, signal, advice, current_time,
                                                            flo_pre_contract_balance)

        # Contract status check
        if boo_is_contract_open:
            str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter = statuscheck.contractstatus(flo_pre_contract_balance, int_positive_counter)

            if new_profit_balance != flo_pre_contract_balance:
                boo_is_contract_open = False
                flo_pre_contract_balance = new_profit_balance
                ai_logging.log_contract_end(contract_id, str_contract_end)

        # Limit candle list to last 30 entries
        if len(lst_fetched_candles) > 30:
            lst_fetched_candles.pop()

        """ mainwindow.update_screen(lst_fetched_candles, macd, signal, advice, start_balance, start_datetime,
                                 int_number_of_candles, market_condition, direction, flo_contract_size,
                                 int_stop_loss_distance, int_this_contract_profit, int_positive_counter)"""
        #  Update the PyQt5 UI dynamically
        ui_window.update_ui_signal.emit(
            lst_fetched_candles,
            macd,
            signal,
            advice,
            start_balance,
            start_datetime,
            int_number_of_candles,
            market_condition,
            direction,
            flo_contract_size,
            int_stop_loss_distance,
            int_this_contract_profit,
            int_positive_counter
        )
        sleep_time.sleep(30)  # Wait 3 minutes before next loop


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
