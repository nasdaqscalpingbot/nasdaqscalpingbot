import csv
import os
import connection

contract_cache = {}
source_data_file = './ml_data/contracts_ai_data.csv'

def is_ai_correct(profit_loss):
    """
    Bepaal of de AI het goed had op basis van het resultaat.
    """
    return profit_loss > 0

def detect_market_condition(candles):
    """
    Bepaal de marktomstandigheden op basis van de laatste candles.
    Retourneert een dictionary met 'trend' en 'volatiliteit'.
    """
    # Voorbeeld: Bepaal trend op basis van de laatste 3 candles
    open_prices = [candle[0] for candle in candles[:3]]
    close_prices = [candle[1] for candle in candles[:3]]

    if close_prices[-1] > open_prices[0]:
        trend = "Uptrend"
    elif close_prices[-1] < open_prices[0]:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    # Voorbeeld: Bepaal volatiliteit op basis van het bereik van de candles
    ranges = [candle[3] - candle[4] for candle in candles[:3]]  # High - Low
    avg_range = sum(ranges) / len(ranges)
    volatiliteit = "High" if avg_range > 0.005 else "Low"  # Pas de drempelwaarde aan

    return {
        'trend': trend,
        'volatiliteit': volatiliteit
    }

def store_ai_decision(rf_decision, rf_confidence, xgb_decision, xgb_confidence, final_decision):
    contract_cache['rf_decision'] = rf_decision
    contract_cache['rf_confidence'] = rf_confidence
    contract_cache['xgb_decision'] = xgb_decision
    contract_cache['xgb_confidence'] = xgb_confidence
    contract_cache['final_decision'] = final_decision

def log_contract_start(lst_fetched_candles, macd, signal, current_time, flo_pre_contract_balance):
    contract_id = connection.get_open_position()
    if not contract_id:
        print("Error: No contract ID found.")
        return None

    # Bereken marktomstandigheden bij openen
    market_condition_start = detect_market_condition(lst_fetched_candles)

    # Cache contract data
    contract_cache[contract_id] = {
        'Timestamp': current_time,
        'Balance': flo_pre_contract_balance,
        'Current_candle_open': lst_fetched_candles[0][0],
        'Previous_candle_open': lst_fetched_candles[1][0],
        'Oldest_candle_open': lst_fetched_candles[2][0],
        'MACD_line': "{:.2f}".format(macd),
        'Signal_line': "{:.2f}".format(signal),
        'Market Condition Start': market_condition_start,  # Voeg marktomstandigheden toe
        'RF Decision': contract_cache.get('rf_decision', 'N/A'),
        'RF Confidence': contract_cache.get('rf_confidence', 'N/A'),
        'XGB Decision': contract_cache.get('xgb_decision', 'N/A'),
        'XGB Confidence': contract_cache.get('xgb_confidence', 'N/A'),
        'Final Decision': contract_cache.get('final_decision', 'N/A'),
        'Profit/Loss': 'N/A',
        'Status': 'OPEN'
    }

    # Append to CSV
    try:
        append_contract_to_csv(contract_id)
    except Exception as e:
        print(f"Error writing to CSV: {e}")

    return contract_id


def append_contract_to_csv(contract_id):
    file_exists = os.path.isfile(source_data_file)
    contract_data = contract_cache.get(contract_id)

    if not contract_data:
        print("Error: No contract data found in cache.")
        return

    try:
        with open(source_data_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    'Contract ID', 'Timestamp', 'Balance', 'Current candle open',
                    'Previous candle open', 'Oldest candle open', 'MACD line',
                    'Signal line', 'Market Condition Start', 'Market Condition End',
                    'RF Decision', 'RF Confidence', 'XGB Decision', 'XGB Confidence',
                    'Final Decision', 'Profit/Loss', 'New Balance', 'Status', 'AI Correct'
                ])

            # Veilige manier om market conditions te verwerken
            market_condition_start = "N/A"
            if 'Market Condition Start' in contract_data:
                market_condition_start = f"{contract_data['Market Condition Start'].get('trend', 'N/A')}, {contract_data['Market Condition Start'].get('volatiliteit', 'N/A')}"

            market_condition_end = "N/A"  # Default waarde
            if 'Market Condition End' in contract_data:
                market_condition_end = f"{contract_data['Market Condition End'].get('trend', 'N/A')}, {contract_data['Market Condition End'].get('volatiliteit', 'N/A')}"

            writer.writerow([
                contract_id,
                contract_data.get('Timestamp', 'N/A'),
                contract_data.get('Balance', 'N/A'),
                contract_data.get('Current_candle_open', 'N/A'),
                contract_data.get('Previous_candle_open', 'N/A'),
                contract_data.get('Oldest_candle_open', 'N/A'),
                contract_data.get('MACD_line', 'N/A'),
                contract_data.get('Signal_line', 'N/A'),
                market_condition_start,
                market_condition_end,
                contract_data.get('RF Decision', 'N/A'),
                contract_data.get('RF Confidence', 'N/A'),
                contract_data.get('XGB Decision', 'N/A'),
                contract_data.get('XGB Confidence', 'N/A'),
                contract_data.get('Final Decision', 'N/A'),
                contract_data.get('Profit/Loss', 'N/A'),
                contract_data.get('New Balance', 'N/A'),
                contract_data.get('Status', 'N/A'),
                contract_data.get('AI Correct', 'N/A')
            ])
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def log_contract_end(contract_id, str_contract_end, int_this_contract_profit, new_profit_balance, lst_fetched_candles):
    print("Log contract end")
    print(contract_id, str_contract_end, int_this_contract_profit, new_profit_balance)
    if contract_id in contract_cache:
        market_condition_end = detect_market_condition(lst_fetched_candles)

        # Update de cache
        contract_cache[contract_id]['Status'] = str_contract_end
        contract_cache[contract_id]['Profit/Loss'] = int_this_contract_profit
        contract_cache[contract_id]['New Balance'] = new_profit_balance
        contract_cache[contract_id]['Market Condition End'] = market_condition_end

        ai_correct = is_ai_correct(int_this_contract_profit)
        contract_cache[contract_id]['AI Correct'] = ai_correct

        update_csv_contract(contract_id)

def update_csv_contract(contract_id):
    rows = []
    contract_data = contract_cache.get(contract_id)

    if not contract_data:
        print("Error: No contract data found in cache.")
        return

    try:
        # Read all rows from the existing CSV
        with open(source_data_file, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Save headers
            rows = list(reader)  # Store all rows in memory

        if rows:
            # Update the last row with the latest contract data
            last_row = rows[-1]
            last_row[9] = contract_data.get('Market Condition End', 'N/A')
            last_row[14] = contract_data['Profit/Loss']
            last_row[15] = contract_data['New Balance']
            last_row[16] = contract_data['Status']
            last_row[17] = contract_data.get('AI Correct', 'N/A')

        # Rewrite the CSV with updated data
        with open(source_data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers back
            writer.writerows(rows)  # Write updated rows
    except Exception as e:
        print(f"Error updating CSV: {e}")

def update_errorlog(retry_datetime, endpoint, retry_counter):
    file_exists = os.path.isfile('error_log.csv')
    with open('error_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writerow(['Date/time', 'Endpoint', 'Retry counter'])

        # Append the contract data to the CSV
        writer.writerow([retry_datetime, endpoint, retry_counter])