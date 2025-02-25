# ========================================== Imports ===================================================================
import json
import connection



# This function calculate the contract details
def calculate_position_sizes(flo_pre_contract_balance):
    flo_basic_contract_size = round((flo_pre_contract_balance - 300) / 1000, 1)                     # Contract size
    # int_contract_take_profit = round(S_SESSION.flo_pre_contract_balance * 0.02, 0)                            # Take profit size, 2% of the saldo
    # int_contract_stop_loss = round(S_SESSION.flo_pre_contract_balance * 0.01, 0)                              # Stop loss size, 1% of the saldo
    return flo_basic_contract_size

def calculate_trailing_stop_loss(lst_fetched_candles: list[list[float]], multiplier: int = 25, min_stop_loss: int = 20) -> float:
    """
    Calculate stop-loss distance based on recent open-close differences.
    """
    recent_candles = lst_fetched_candles[:10]
    open_close_differences = [abs(candle[0] - candle[1]) for candle in recent_candles]
    # Calculate the average difference
    avg_diff = sum(open_close_differences) / len(open_close_differences) if open_close_differences else 0
    # Scale the stop-loss and ensure it's above the minimum
    print("avg_diff", avg_diff)
    stop_loss_distance = avg_diff * multiplier
    if stop_loss_distance > min_stop_loss:
        int_stop_loss_distance = stop_loss_distance
    else:
        int_stop_loss_distance = min_stop_loss
    print(f"Calculated Stop-Loss Distance: {int_stop_loss_distance}")
    return int_stop_loss_distance


# This function handle the sequence of setting up a new contract
def open_new_contract(lst_fetched_candles, direction):
    account_information = connection.account_details()  # Get the current account details/balance
    flo_pre_contract_balance = account_information['accounts'][0]['balance']['balance']  # Store the current account balance
    flo_contract_size = calculate_position_sizes(flo_pre_contract_balance)                                                                      # Starts with calculating the contract values
    int_stop_loss_distance = calculate_trailing_stop_loss(lst_fetched_candles)
    connection.active_account()                                                                                                    # Activate the active account
    payload = json.dumps({                                                                                              # Prepare the position request
        "epic": "US100",                                                                                                # Set the market
        "direction": direction,                                                                        # Set the direction buy or sell
        "size": flo_contract_size,                                                                      # Set the contracts size
        "level": 20,                                                                                                    # Set the level to 20, required, don't know what is does
        "type": "LIMIT",                                                                                                # Make it a limit order
        # "stopAmount": S_SESSION.int_contract_stop_loss,                                                                 # Set the stop-loss amount
        # "profitAmount": S_SESSION.int_contract_take_profit                                                              # Set the take profit amount
        "stopDistance": int_stop_loss_distance,
        # "profitDistance": 30,
        "trailingStop": True
    })
    connection.create_position(payload)                                                                                            # Create the actual contract (connection.py)
    return direction, flo_contract_size, int_stop_loss_distance
