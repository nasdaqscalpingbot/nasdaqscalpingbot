# ============== Imports =====================================
import tkinter as tk
from datetime import datetime
from connection import account_details

# ============== Variables =====================================

def start_button_clicked():
    pass


window = tk.Tk()
window.title('Scalping AI')
window.geometry('1024x800')
window.config(bg ='whitesmoke')

#Title
title_label = tk.Label(window, text='Nasdaq scalping bot 2.0a', font=("Helvetica", 24), bg='whitesmoke', fg='black')
title_label.pack(pady=(50, 10))  # Adds 50 pixels above and 10 pixels below

#start button
start_button = tk.Button(window, text="Start", font=("Helvetica", 14, "bold"),
    bg='#66ffff', fg='black', relief='raised', bd=3 )
start_button.pack(pady=20)

# Horizontal divider
horizontal_line = tk.Label(window, text="─" * 90, font=("Helvetica", 12), bg='whitesmoke', fg='black')
horizontal_line.pack(pady=20)

frame = tk.Frame(window)
frame.pack(fill='both', expand=True)
frame.configure(bg='whitesmoke')
widgetbackground = 'whitesmoke'
widgetforeground = '#000000'
placeholderbackground = 'whitesmoke'

frame.columnconfigure(0, weight=2)  # Margin left
frame.columnconfigure(1, weight=1)  # Label2 column 1
frame.columnconfigure(2, weight=1)  # Placeholders 1
frame.columnconfigure(3, weight=1)  # Margin middle
frame.columnconfigure(4, weight=1)  # Labels column 2
frame.columnconfigure(5, weight=1)  # Placeholders 2
frame.columnconfigure(6, weight=1)  # Margin right


frame.rowconfigure(0, weight=1)
frame.rowconfigure(1, weight=1)
frame.rowconfigure(2, weight=1)
frame.rowconfigure(3, weight=1)
frame.rowconfigure(4, weight=1)
frame.rowconfigure(5, weight=1)
frame.rowconfigure(6, weight=1)
frame.rowconfigure(7, weight=1)
frame.rowconfigure(8, weight=1)
frame.rowconfigure(9, weight=1)
frame.rowconfigure(10, weight=1)
frame.rowconfigure(11, weight=4)

font_family = "Helvetica"
font_size = 16


# Current time
timeupdate_label = tk.Label(frame, text='Last update:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
timeupdate_label.grid(row=0, column=1, sticky='w')
placeholder_time = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_time.grid(row=0, column=2, sticky='we')

# ------------------- Columns left, giving advice information ----------------------------------------------------------

# Current number of candles
number_of_candles_label = tk.Label(frame, text='Number of candles:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
number_of_candles_label.grid(row=1, column=1, sticky='w')
placeholder_candles = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_candles.grid(row=1, column=2, sticky='we')

# Current candle
current_candle_open_label = tk.Label(frame, text='Current candle open:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
current_candle_open_label.grid(row=2, column=1, sticky='w')
placeholder_current_candle_open = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_current_candle_open.grid(row=2, column=2, sticky='we')

# Previous candle
previous_candle_open_label = tk.Label(frame, text='Previous candle open:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
previous_candle_open_label.grid(row=3, column=1, sticky='w')
placeholder_previous_candle_open = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_previous_candle_open.grid(row=3, column=2, sticky='we')

# Oldest candle
oldest_candle_open_label = tk.Label(frame, text='Oldest candle open:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
oldest_candle_open_label.grid(row=4, column=1, sticky='w')
placeholder_oldest_candle_open = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_oldest_candle_open.grid(row=4, column=2, sticky='we')

# Current ema, when available
macd_label = tk.Label(frame, text='Current macd_line:', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
macd_label.grid(row=5, column=1, sticky='w')
placeholder_macd = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_macd.grid(row=5, column=2, sticky='we')

# current signal_line
signal_line_label = tk.Label(frame, text='Current signal line:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
signal_line_label.grid(row=6, column=1, sticky='w')
placeholder_signal_line = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_signal_line.grid(row=6, column=2, sticky='we')


# current histogram
histogram_label = tk.Label(frame, text='Current histogram:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
histogram_label.grid(row=7, column=1, sticky='w')
placeholder_histogram = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_histogram.grid(row=7, column=2, sticky='we')

# Current given advice
advice_label = tk.Label(frame, text='Current advice:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
advice_label.grid(row=8, column=1, sticky='w')
placeholder_advice = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_advice.grid(row=8, column=2, sticky='we')

# ------------------- Columns right, contract information ----------------------------------------------------------

# Current contract, the contract that is actually active or was the most recent active contract
current_contract_label = tk.Label(frame, text='Current contract:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
current_contract_label.grid(row=1, column=4, sticky='w')
placeholder_current_contract = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_current_contract.grid(row=1, column=5, sticky='we')

# Contract size
contract_size_label = tk.Label(frame, text='Contract size:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
contract_size_label.grid(row=2, column=4, sticky='w')
placeholder_contract_size = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_contract_size.grid(row=2, column=5, sticky='we')

# Current profit
current_profit_label = tk.Label(frame, text='Current profit/loss:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
current_profit_label.grid(row=3, column=4, sticky='w')
placeholder_current_profit = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_current_profit.grid(row=3, column=5, sticky='we')

# Take profit
take_profit_label = tk.Label(frame, text='Take profit:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
take_profit_label.grid(row=4, column=4, sticky='w')
placeholder_take_profit = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_take_profit.grid(row=4, column=5, sticky='we')

# Stop loss
stop_loss_label = tk.Label(frame, text='Stop loss:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
stop_loss_label.grid(row=5, column=4, sticky='w')
placeholder_stop_loss = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_stop_loss.grid(row=5, column=5, sticky='we')


# Positive profit counter
positive_PL_label = tk.Label(frame, text='Positive P/L counter:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
positive_PL_label.grid(row=6, column=4, sticky='w')
placeholder_positive_PL = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_positive_PL.grid(row=6, column=5, sticky='we')

# Negative profit counter
negative_PL_label = tk.Label(frame, text='Negative P/L counter:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
negative_PL_label.grid(row=7, column=4, sticky='w')
placeholder_negative_PL = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_negative_PL.grid(row=7, column=5, sticky='we')

# Stat date time
starttime_label = tk.Label(frame, text='Started at:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
starttime_label.grid(row=8, column=4, sticky='w')
placeholder_starttime = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_starttime.grid(row=8, column=5, sticky='we')

# Total profit
total_profit_label = tk.Label(frame, text='Total profit since start:', font=(font_family, font_size), fg= widgetforeground, bg= widgetbackground)
total_profit_label.grid(row=9, column=4, sticky='w')
placeholder_total_profit = tk.Label(frame, text='', font=(font_family, font_size), fg= widgetforeground, bg= placeholderbackground)
placeholder_total_profit.grid(row=9, column=5, sticky='we')


def update_screen(S_SESSION):
    window_bg = 'whitesmoke'
    labels_bg = "whitesmoke"
    widget_bg = 'whitesmoke'
    widget_fg = 'black'
    now = datetime.now()  # Get the current date and time
    str_current_datetime = now.strftime("%d-%m-%Y %H:%M:%S")  # Format as DD-MM-YYYY HH:MM:SS
    #S_SESSION.str_given_advice = "BUY"
    # Set default colors based on the condition
    if S_SESSION.str_current_contract == "BUY":
        window_bg = '#33ff33'
        labels_bg = "#33ff33"
        widget_bg = 'whitesmoke'
        widget_fg = 'black'
    elif S_SESSION.str_current_contract == "SELL":
        window_bg = '#ff8566'
        labels_bg = "#ff8566"
        widget_bg = 'whitesmoke'
        widget_fg = 'black'
    # Update window and frame background color
    window.configure(bg=window_bg)
    frame.configure(bg=window_bg)

    account_information = account_details()
    S_SESSION.flo_current_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
    total_profit = S_SESSION.flo_current_balance - S_SESSION.flo_start_balance

    # Update text and the colors in placeholders
    # Define a dictionary to store labels and corresponding placeholder updates
    label_placeholder_pairs = {
        "time": (timeupdate_label, placeholder_time, str_current_datetime),
        "int_candles": (number_of_candles_label, placeholder_candles, S_SESSION.int_number_of_candles),
        "int_current_candle_open": (current_candle_open_label, placeholder_current_candle_open, S_SESSION.lis_fetched_candles[0][0]),
        "int_previous_candle_open": (previous_candle_open_label, placeholder_previous_candle_open, S_SESSION.lis_fetched_candles[1][0]),
        "int_oldest_candle_open": (oldest_candle_open_label, placeholder_oldest_candle_open, S_SESSION.lis_fetched_candles[2][0]),
        "macd_line": (macd_label, placeholder_macd, S_SESSION.flo_macd_line),
        "signal_line": (signal_line_label, placeholder_signal_line, S_SESSION.flo_signal_line),
        "histogram": (histogram_label, placeholder_histogram, S_SESSION.flo_histogram),
        "str_current_contract": ( current_contract_label, placeholder_current_contract, S_SESSION.str_current_contract),
        "str_advice": (advice_label, placeholder_advice, S_SESSION.str_given_advice),
        "flo_contract_size": (contract_size_label, placeholder_contract_size, S_SESSION.flo_basic_contract_size),
        "int_take_profit": (take_profit_label, placeholder_take_profit, "$" + str(S_SESSION.int_contract_take_profit)),
        "int_stop_loss": (stop_loss_label, placeholder_stop_loss,"$" + str(S_SESSION.int_contract_stop_loss)),
        "int_current_profit": (current_profit_label, placeholder_current_profit, "€" + str(S_SESSION.int_this_contract_profit)),
        "positive_PL": (positive_PL_label, placeholder_positive_PL,str(S_SESSION.int_positive_counter)+"/15"),
        "negative_PL": (negative_PL_label, placeholder_negative_PL, str(S_SESSION.int_negative_counter) + "/9"),
        "starttime": (starttime_label, placeholder_starttime, S_SESSION.str_start_datetime),
        "int_total_profit": (total_profit_label, placeholder_total_profit, "€" + str(round(total_profit,2)))
    }

    # Loop through each label and placeholder to apply the same settings
    for key, (label, placeholder, text) in label_placeholder_pairs.items():
        placeholder.config(text=text, fg=widget_fg, bg=widget_bg)
        label.config(fg=widget_fg, bg=labels_bg)


