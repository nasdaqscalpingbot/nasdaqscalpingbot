import csv
import os
import connection

contract_cache = {}
source_data_file = './ml_data/contracts_data.csv'

def log_contract_start(lst_fetched_candles, macd, signal, advice, current_time, flo_pre_contract_balance):
    contract_id = connection.get_open_position()

    # Cache contract data in memory
    if advice == "BUY":
        advice = 1
    elif advice == "SELL":
        advice = 2
    contract_cache[contract_id] = {
        'Timestamp':                    current_time,
        'Balance':                      flo_pre_contract_balance,
        'Current_candle_open':          lst_fetched_candles[0][0],
        'Current_candle_close':         lst_fetched_candles[0][1],
        'Previous_candle_open':         lst_fetched_candles[1][0],
        'Previous_candle_close':        lst_fetched_candles[1][1],
        'Oldest_candle_open':           lst_fetched_candles[2][0],
        'Oldest_candle_close':          lst_fetched_candles[2][1],
        'MACD_line': "{:.2f}".format(macd),
        'Signal_line': "{:.2f}".format(signal),
        'Buy/Sell':                     advice,
        'Profit/Loss': 'N/A',
        'Status': 'OPEN'
    }
    append_contract_to_csv(contract_id)
    return contract_id

def append_contract_to_csv(contract_id):
    file_exists = os.path.isfile(source_data_file)

    with open(source_data_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writerow(['Contract ID', 'Timestamp', 'Balance',
                             'Current candle open', 'Current candle close',
                             'Previous candle open', 'Previous candle close',
                             'Oldest candle open', 'Oldest candle close',
                             'MACD line', 'Signal_line', 'Status', 'Buy/Sell'])
        # Append the contract data to the CSV
        contract_data = contract_cache[contract_id]
        writer.writerow([
            contract_id,
            contract_data['Timestamp'],
            contract_data['Balance'],
            contract_data['Current_candle_open'],
            contract_data['Current_candle_close'],
            contract_data['Previous_candle_open'],
            contract_data['Previous_candle_close'],
            contract_data['Oldest_candle_open'],
            contract_data['Oldest_candle_close'],
            contract_data['MACD_line'],
            contract_data['Signal_line'],
            contract_data['Status'],
            contract_data['Buy/Sell']
        ])

def log_contract_end(contract_id, str_contract_end):
    # Update the cached data
    if contract_id in contract_cache:
        contract_cache[contract_id]['Status'] = str_contract_end  # e.g., "CLOSED" or "STOPPED"

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