import connection

def calculate_new_profit_balance(flo_pre_contract_balance):
    account_information = connection.account_details()  # Get the current account information
    int_after_contract_balance = round(account_information['accounts'][0]['balance']['balance'], 2)           # Get the current balance
    int_this_contract_profit = round((int_after_contract_balance - flo_pre_contract_balance), 2)  # Calculate the profit or loss
    return int_this_contract_profit


# This function handles the final steps after a contract is closed

def contractstatus(flo_pre_contract_balance, int_positive_counter):
    """Check the status of the open contract and close it if conditions are met."""
    str_contract_end = ""
    new_profit_balance = flo_pre_contract_balance
    account_information = connection.account_details()
    int_this_contract_profit = account_information['accounts'][0]['balance']['profitLoss']

    # Contract closed (either trailing stop or take profit)
    if int_this_contract_profit == 0.0:
        new_profit_balance = calculate_new_profit_balance(flo_pre_contract_balance)
        str_contract_end = "Trailing Stop" if int_this_contract_profit < 0 else "Take Profit"
        int_positive_counter = 0
        return str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter

    # Contract still positive
    elif int_this_contract_profit > 0:
        int_positive_counter += 1  # Increment the positive counter
        if int_positive_counter >= 4:  # Close after 21 minutes of profit
            str_contract_id = connection.get_open_position()
            connection.close_position(str_contract_id)
            new_profit_balance = calculate_new_profit_balance(flo_pre_contract_balance)
            str_contract_end = "Take Profit (20 minutes)"
        return str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter

    # Contract still open but not positive
    int_positive_counter = 0  # Reset positive counter if not positive
    return str_contract_end, new_profit_balance, int_this_contract_profit, int_positive_counter

