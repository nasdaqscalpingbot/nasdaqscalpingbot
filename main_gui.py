# startloop.py
import sys
import threading
from datetime import datetime, time
import time as sleep_time

from PyQt5.QtWidgets import QApplication

import connection
import contracts
import market
import strategy
import statuscheck
import ai_logging
# from session import S_SESSION

from interface.scalpingbotview import ScalpingbotView


lst_fetched_candles = []

def start_loop(ui_window):
    boo_is_contract_open = False
    contract_id = ""
    str_contract_id = ""
    macd = 0.0
    signal = 0.0
    advice = "NEUTRAL"


    connection.create_new_session()
    account_information = connection.account_details()
    start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
    flo_pre_contract_balance = start_balance

    now = datetime.now()
    start_datetime = now.strftime("%d-%m-%Y %H:%M:%S")

    while True:
        current_time = datetime.now().time()

        if current_time >= time(22, 50) or current_time <= time(0, 10):
            sleep_time.sleep(5400)  # Market closed, wait 1.5 hours
            connection.create_new_session()
            account_information = connection.account_details()
            start_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
        else:
            lst_fetched_candles.insert(0, market.fetch_current_market_info())

        int_candle_counter = len(lst_fetched_candles)

        if int_candle_counter > 25 and not boo_is_contract_open:
            market_condition = market.detect_market_condition(lst_fetched_candles)
            if market_condition == "NORMAL":
                advice, macd, signal = strategy.strategycheck(lst_fetched_candles)
                if advice != "HOLD":
                    flo_pre_contract_balance = contracts.open_new_contract(lst_fetched_candles, advice)
                    boo_is_contract_open = True
                    contract_id = ai_logging.log_contract_start(lst_fetched_candles, macd, signal, advice, current_time, flo_pre_contract_balance)

        if boo_is_contract_open:
            str_contract_end, new_profit_balance = statuscheck.contractstatus(flo_pre_contract_balance)
            if new_profit_balance != flo_pre_contract_balance:
                ai_logging.log_contract_end(contract_id, str_contract_end, new_profit_balance)

        if int_candle_counter > 30:
            lst_fetched_candles.pop()

        # ✅ Update PyQt5 UI dynamically
        ui_window.update_ui_signal.emit(
            lst_fetched_candles,  # list
            macd,  # float
            signal,  # float
            advice,  # str
            datetime.now().strftime("%H:%M:%S"),  # str
            flo_pre_contract_balance  # float
        )

        sleep_time.sleep(180)  # Wait 3 minutes before next loop


def main():
    app = QApplication(sys.argv)
    window = ScalpingbotView()  # Initialize PyQt5 UI

    # Start the bot loop in a separate thread
    bot_thread = threading.Thread(target=start_loop, args=(window,))
    bot_thread.daemon = True  # Ensures the thread closes when the app exits
    bot_thread.start()

    window.show()
    sys.exit(app.exec_())  # Start PyQt5 event loop


# ========================================== Main start ================================================================
if __name__ == "__main__":
    main()
