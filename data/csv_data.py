import csv
import os

contract_cache = {}
source_data_file = './ml_data/contract_data.csv'

def log_contract_start(S_SESSION):
    contract_id = S_SESSION.str_contract_id if S_SESSION.str_contract_id is not None else '99'
    # Cache contract data in memory
    if S_SESSION.str_given_advice == "BUY":
        advice = 1
    elif S_SESSION.str_given_advice == "SELL":
        advice = 2
    else:
        advice = 0
    contract_cache[contract_id] = {
        'Timestamp':                    S_SESSION.current_candle_time,
        'Balance':                      S_SESSION.flo_pre_contract_balance,
        'Current_candle_open':          S_SESSION.current_candle_open,
        'Current_candle_close':         S_SESSION.current_candle_close,
        'Current_candle_high':          S_SESSION.current_candle_high,
        'Current_candle_low':           S_SESSION.current_candle_low,
        'Current_candle_netchange':     S_SESSION.current_candle_netChange,
        'Previous_candle_open':         S_SESSION.previous_candle_open,
        'Previous_candle_close':        S_SESSION.previous_candle_close,
        'Previous_candle_high':         S_SESSION.previous_candle_high,
        'Previous_candle_low':          S_SESSION.previous_candle_low,
        'Previous_candle_netchange':    S_SESSION.previous_candle_netChange,
        'Oldest_candle_open':           S_SESSION.oldest_candle_open,
        'Oldest_candle_close':          S_SESSION.oldest_candle_close,
        'Oldest_candle_high':           S_SESSION.oldest_candle_high,
        'Oldest_candle_low':            S_SESSION.oldest_candle_low,
        'Oldest_candle_netchange':      S_SESSION.oldest_candle_netChange,
        'MACD_line': "{:.2f}".format(S_SESSION.flo_macd_line) if S_SESSION.flo_macd_line != 0.0 else 'N/A',
        'Signal_line': "{:.2f}".format(S_SESSION.flo_signal_line) if S_SESSION.flo_signal_line != 0.0 else 'N/A',
        'Histogram': "{:.2f}".format(S_SESSION.flo_histogram) if S_SESSION.flo_histogram != 0.0 else 'N/A',
        'Buy/Sell': advice,
        'Profit/Loss': S_SESSION.int_this_contract_profit if S_SESSION.int_this_contract_profit != 0.0 else 'N/A',
        'Status': 'OPEN'
    }
    append_contract_to_csv(contract_id)

def append_contract_to_csv(contract_id):
    file_exists = os.path.isfile(source_data_file)

    with open(source_data_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writerow(['Contract ID', 'Timestamp', 'Balance', 'Current candle open', 'Current candle close',
                             'Current candle high', 'Current candle low', 'Current candle netchange',
                             'Previous candle open', 'Previous candle close',
                             'Previous candle high', 'Previous candle low', 'Previous candle netchange',
                             'Oldest candle open', 'Oldest candle close',
                             'Oldest candle high', 'Oldest candle low', 'Oldest candle netchange',
                             'MACD line', 'Signal_line', 'Histogram', 'Status', 'Buy/Sell'])
        # Append the contract data to the CSV
        contract_data = contract_cache[contract_id]
        writer.writerow([
            contract_id,
            contract_data['Timestamp'],
            contract_data['Balance'],
            contract_data['Current_candle_open'],
            contract_data['Current_candle_close'],
            contract_data['Current_candle_high'],
            contract_data['Current_candle_low'],
            contract_data['Current_candle_netchange'],
            contract_data['Previous_candle_open'],
            contract_data['Previous_candle_close'],
            contract_data['Previous_candle_high'],
            contract_data['Previous_candle_low'],
            contract_data['Previous_candle_netchange'],
            contract_data['Oldest_candle_open'],
            contract_data['Oldest_candle_close'],
            contract_data['Oldest_candle_high'],
            contract_data['Oldest_candle_low'],
            contract_data['Oldest_candle_netchange'],
            contract_data['MACD_line'],
            contract_data['Signal_line'],
            contract_data['Histogram'],
            contract_data['Status'],
            contract_data['Buy/Sell']
        ])

def log_contract_end(contract_id, final_profit_loss, final_status):
    # Update the cached data
    if contract_id in contract_cache:
        contract_cache[contract_id]['Status'] = final_status  # e.g., "CLOSED" or "STOPPED"

        # Reload and modify CSV data
        update_csv_contract(contract_id)

def update_csv_contract(contract_id):
    rows = []

    # Read all rows from the existing CSV
    with open(source_data_file, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Save headers
        for row in reader:
            if row[0] == contract_id:  # Update the row with the matching contract ID
                row[21] = contract_cache[contract_id]['Status']  # Update Status
            rows.append(row)

    # Rewrite the CSV with updated data
    with open(source_data_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers back
        writer.writerows(rows)  # Write updated rows

def update_errorlog(retry_datetime, endpoint, retry_counter):
    file_exists = os.path.isfile('error_log.csv')
    with open('error_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writerow(['Date/time', 'Endpoint', 'Retry counter'])

        # Append the contract data to the CSV
        writer.writerow([retry_datetime, endpoint, retry_counter])