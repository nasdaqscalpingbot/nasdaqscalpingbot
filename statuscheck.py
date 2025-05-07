import connection

def calculate_new_profit_balance(flo_pre_contract_balance):
    account_information = connection.account_details()  # Get the current account information
    int_after_contract_balance = round(account_information['accounts'][0]['balance']['balance'], 2)           # Get the current balance
    int_this_contract_profit = round((int_after_contract_balance - flo_pre_contract_balance), 2)  # Calculate the profit or loss
    return int_this_contract_profit


# This function handles the final steps after a contract is closed

from typing import Tuple

def contractstatus(flo_pre_contract_balance: float, int_positive_counter: int) -> Tuple[str, float, float, int]:
    """
    Check the status of the open contract and close it if conditions are met.

    Args:
        flo_pre_contract_balance (float): The balance before the contract was opened.
        int_positive_counter (int): A counter tracking how long the contract has been in profit.

    Returns:
        tuple: A tuple containing:
            - contract_end (str): Reason for contract closure (e.g., "Trailing Stop", "Take Profit").
            - new_profit_balance (float): Updated balance after contract closure.
            - this_contract_profit (float): Profit/loss of the current contract.
            - positive_counter (int): Updated positive counter.
    """
    contract_end = ""
    new_profit_balance = flo_pre_contract_balance

    try:
        account_information = connection.account_details()
        this_contract_profit = account_information['accounts'][0]['balance']['profitLoss']
    except (KeyError, TypeError, ConnectionError) as e:
        print(f"Error retrieving account information: {e}")
        return contract_end, new_profit_balance, 0.0, int_positive_counter

    # Contract closed (either trailing stop or take profit)
    if this_contract_profit == 0.0:
        new_profit_balance = calculate_new_profit_balance(flo_pre_contract_balance)
        contract_end = "Trailing Stop" if this_contract_profit < 0 else "Take Profit"
        int_positive_counter = 0

    # # Contract still positive
    # elif this_contract_profit > 0:
    #     int_positive_counter += 1  # Increment the positive counter
    #     if int_positive_counter >= 4:  # Close after 21 minutes of profit
    #         contract_id = connection.get_open_position()
    #         connection.close_position(contract_id)
    #         new_profit_balance = calculate_new_profit_balance(flo_pre_contract_balance)
    #         contract_end = "Take Profit (20 minutes)"

    # Contract still open but not positive
    return contract_end, new_profit_balance, this_contract_profit, int_positive_counter

