import pandas as pd

# Load the CSV file
file_path = 'contract_data_copy.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Initialize violation counters, logs, and cost
buy_violations = []
sell_violations = []
total_buy_cost = 0
total_sell_cost = 0


# Define validation function
def validate_signal_and_cost(row):
    global total_buy_cost, total_sell_cost
    macd = row['MACD line']
    signal = row['Signal_line']
    histogram = row['Histogram']
    signal_type = row['B/S']
    profit_loss = row['Profit/Loss'] if not pd.isna(row['Profit/Loss']) else 0  # Handle missing P/L

    # Candle values
    current_open = row['Current candle open']
    previous_open = row['Previous candle open']
    oldest_open = row['Oldest candle open']

    # Check conditions for BUY
    if signal_type == 'BUY':
        candle_condition = (
                (current_open > previous_open + 2 and previous_open > oldest_open + 2) or
                (current_open > previous_open + 10)
        )
        macd_condition = macd > signal and histogram > 1.5
        if not (candle_condition and macd_condition):
            cause = []
            if not candle_condition:
                cause.append("Candle condition failed")
            if not macd_condition:
                if macd <= signal:
                    cause.append("MACD <= Signal")
                if histogram <= 3:
                    cause.append("Histogram <= 3")
            buy_violations.append({'Index': row.name, 'Cause': ', '.join(cause), 'Profit/Loss': profit_loss})
            total_buy_cost += profit_loss  # Add actual profit/loss to the total cost of BUY violations

    # Check conditions for SELL
    elif signal_type == 'SELL':
        candle_condition = (
                (current_open < previous_open - 2 and previous_open < oldest_open - 2) or
                (current_open < previous_open - 10)
        )
        macd_condition = macd < signal and histogram < -1.5
        if not (candle_condition and macd_condition):
            cause = []
            if not candle_condition:
                cause.append("Candle condition failed")
            if not macd_condition:
                if macd >= signal:
                    cause.append("MACD >= Signal")
                if histogram >= -3:
                    cause.append("Histogram >= -3")
            sell_violations.append({'Index': row.name, 'Cause': ', '.join(cause), 'Profit/Loss': profit_loss})
            total_sell_cost += profit_loss  # Add actual profit/loss to the total cost of SELL violations


def main():
    # Apply validation to each row
    data.apply(validate_signal_and_cost, axis=1)

    # Print results
    print(f"Total BUY Violations: {len(buy_violations)}")
    print(f"Total BUY Violation Cost: ${total_buy_cost:.2f}")
    for violation in buy_violations:
        print(f"Row {violation['Index']} - Cause: {violation['Cause']}, Profit/Loss: {violation['Profit/Loss']}")

    print(f"\nTotal SELL Violations: {len(sell_violations)}")
    print(f"Total SELL Violation Cost: ${total_sell_cost:.2f}")
    for violation in sell_violations:
        print(f"Row {violation['Index']} - Cause: {violation['Cause']}, Profit/Loss: {violation['Profit/Loss']}")

if __name__ == "__main__":
     main()