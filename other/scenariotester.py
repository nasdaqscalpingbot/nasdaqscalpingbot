import csv
from datetime import datetime, timedelta

intAdvice = 0
candle_threshold = 5
quick_candle_threshold = 10
histogram_threshold = 3.0

def fetch_all_candles():
    all_candles = []
    row_index = 0
    with open('../ml_data/contract_data.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(row['Signal_line'] != "N/A" and row['Histogram'] != "N/A" and row['MACD line'] != "N/A"):
                current_advice = row['B/S']
                macd = float(row['MACD line'])
                signal = float(row['Signal_line'])
                histogram = float(row['Histogram'])
                current_open = int(row['Current candle open'])
                previous_open = int(row['Previous candle open'])
                oldest_open = int(row['Oldest candle open'])
                contract_advice(oldest_open, previous_open, current_open, macd, signal, histogram, current_advice)



def buy_candle_check(current_candle, previous_candle, oldest_candle):
    if (current_candle - previous_candle) > candle_threshold and (previous_candle - oldest_candle) > candle_threshold:
        return True
    else:
        return False

def sell_candle_check(current_candle, previous_candle, oldest_candle):
    if (previous_candle - current_candle) > candle_threshold and (oldest_candle - previous_candle) > candle_threshold:
        return True
    else:
        return False


def buy_macd_evaluation(macd_line, signal_line, histogram_threshold):
    if macd_line != 0.0 and signal_line != 0.0:
        if macd_line > signal_line and abs(macd_line - signal_line) > histogram_threshold:
            return True
    return False


def sell_macd_evaluation(macd_line, signal_line, histogram_threshold):
    if macd_line != 0.0 and signal_line != 0.0:
        if macd_line < signal_line and abs(signal_line - macd_line) > histogram_threshold:
            return True
    return False

def quick_buy_check(current_candle, previous_candle):
    if (current_candle - previous_candle) > quick_candle_threshold:
        return True
    else:
        return False

def quick_sell_check(current_candle, previous_candle):
    if (previous_candle - current_candle) > quick_candle_threshold:
        return True
    else:
        return False

def contract_advice(oldest_candle, previous_candle, current_candle, macd_line, signal_line, histogram, current_advice):
    global intAdvice
    """
    Determines BUY/SELL/HOLD advice based on EMA, MACD, and candle patterns.

    :param oldest_candle: Value of the oldest candle
    :param previous_candle: Value of the previous candle
    :param current_candle: Value of the current candle
    :param candle_diff: Difference threshold for candle comparison
    :param macd: MACD histogram value
    :param macd_line: Current MACD line value
    :param signal_line: Current signal line value
    :param ema_threshold: Optional EMA difference threshold
    :param macd_threshold: Optional MACD difference threshold
    :return: "BUY", "SELL", or "HOLD"
    """

    # Begin logic evaluation
    advice = "HOLD"
    boo_buy_candle = False
    boo_sell_candle = False
    boo_quick_buy = False
    boo_quick_sell = False
    boo_buy_macd = False
    boo_sell_macd = False


    boo_buy_candle = buy_candle_check(current_candle, previous_candle, oldest_candle)
    boo_sell_candle = sell_candle_check(current_candle, previous_candle, oldest_candle)
    boo_quick_buy = quick_buy_check(current_candle, previous_candle)
    boo_quick_sell = quick_sell_check(current_candle, previous_candle)
    boo_buy_macd = buy_macd_evaluation(macd_line, signal_line, histogram_threshold)
    boo_sell_macd = sell_macd_evaluation(macd_line, signal_line, histogram_threshold)

    if (boo_buy_candle and boo_buy_macd) or boo_quick_buy:
        advice = "BUY"
    elif (boo_sell_candle and boo_sell_macd) or boo_quick_sell:
        advice = "SELL"



    # Log results for debugging
        intAdvice +=1
        print(intAdvice)
        print(f"Advice now: {advice}")
        print(f"Advice was: {current_advice}")
        print(f"Candles: Current={current_candle}, Previous={previous_candle}, Oldest={oldest_candle}")
        print(f"Histogram: {histogram}, MACD Line={macd_line}, Signal Line={signal_line},  Histogram threshold={histogram_threshold}")

    # Return the decision
    return advice


# def determ_profit(advice, current_candle, future_candles, stop_loss, take_profit):
#     if(advice=="BUY"):
#         stop_loss_value = current_candle - stop_loss
#         take_profit_value = current_candle + take_profit
#     else: # "SELL"
#         stop_loss_value = current_candle + stop_loss
#         take_profit_value = current_candle - take_profit
#
#     for row in future_candles:
#         new_candle = row[2]
#         if new_candle <= stop_loss_value:
#             return "Stop-loss",  row[0] # Return the result and timestamp of the stop-loss candle
#         elif new_candle >= take_profit_value:
#             return "take-profit", row[0]  # Return the result and timestamp of the take-profit candle
#     return "end_candles", 1000
#
#
# def save_results_to_file(scenario, number_of_advice, number_of_buy, buy_success_rate,
#                          number_of_sell, sell_success_rate, number_of_stop_loss,
#                          stop_loss_success_rate, number_of_take_profit,
#                          take_profit_success_rate, total_success_rate, filename="results.txt"):
#     with open(filename, 'a') as file:  # 'a' to append if you want to add to the file
#         # Write results to file
#         file.write(f"Results scenario: {scenario['name']}\n")
#         file.write("Number of advices: " + str(number_of_advice) + "\n")
#         file.write("Number of buy: " + str(number_of_buy) + "\n")
#         file.write("Success rate buy: {:.2f}%\n".format(buy_success_rate))
#         file.write("Number of sell: " + str(number_of_sell) + "\n")
#         file.write("Success rate sell: {:.2f}%\n".format(sell_success_rate))
#         file.write("Number of stop loss: " + str(number_of_stop_loss) + "\n")
#         file.write("Success rate stop loss: {:.2f}%\n".format(stop_loss_success_rate))
#         file.write("Number of take profit: " + str(number_of_take_profit) + "\n")
#         file.write("Success rate take profit: {:.2f}%\n".format(take_profit_success_rate))
#         file.write("Total success rate: {:.2f}%\n".format(total_success_rate))
#         file.write("-" * 80 + "\n")  # a long dash to create a horizontal rule
#         file.write("\n")
#
# def fetch_candles(candle_index, all_candles):
#     # Build the candle to be assesed by the contract advice
#     for row in all_candles:
#         if (row[0] == candle_index):
#             oldest_candle = row[2]
#     for row in all_candles:
#         if (row[0] == candle_index + 1):
#             previous_candle = row[2]
#     for row in all_candles:
#         if (row[0] == candle_index + 2):
#             current_candle = row[2]
#             curent_row = row
#     return [oldest_candle,previous_candle,current_candle], curent_row
#
# def run_scenario(all_candles):
#     for scenario in all_scenarios:
#         candle_time = scenario['candles']
#         time_diff = timedelta(minutes=candle_time)
#
#         # Reset the scenario results
#         number_of_advice = 0
#         number_of_buy = 0
#         number_of_sell = 0
#         number_of_stop_loss = 0
#         number_of_take_profit = 0
#         candle_index = 0
#         # set reset scenario settings
#         ema = 0
#         ema9 = 0
#         macd = 0
#         macd_line = 0
#         signal_line = 0
#         candle_diff = scenario['candle_difference']
#
#         while candle_index < len(all_candles):
#             get_advise_candles = fetch_candles(candle_index,all_candles)
#             if(scenario['ema']>0):
#                 ema = scenario['ema']
#                 ema9 = get_advise_candles[1][3]
#             if(scenario['macd']>0):
#                 macd = scenario['macd']
#                 macd_line = get_advise_candles[1][4]
#                 signal_line = get_advise_candles[1][5]
#
#             advice = contract_advice(get_advise_candles[0][2], get_advise_candles[0][1], get_advise_candles[0][0], candle_diff, ema,ema9,macd,macd_line, signal_line)
#             number_of_advice += 1
#             if(advice=="BUY"):
#                 number_of_buy += 1
#             if(advice == "SELL"):
#                 number_of_sell += 1
#
#             stop_loss = scenario['stop_loss_margin']
#             take_profit = scenario['take_profit_margin']
#             if advice == "BUY" or advice == "SELL":
#                 future_candles = all_candles[candle_index + 1:]
#                 result = determ_profit(advice, get_advise_candles[0][2], future_candles,stop_loss, take_profit)
#                 if(result[0] == "Stop-loss"):
#                     number_of_stop_loss += 1
#                 elif(result[0] == "take-profit"): # take profit
#                     number_of_take_profit += 1
#                 candle_index = result[1]
#             else:
#                 candle_index += 1
#
#         # end while
#         # Calculate success rates for individual advice types
#         buy_success_rate = (number_of_buy / number_of_advice) * 100 if number_of_advice > 0 else 0
#         sell_success_rate = (number_of_sell / number_of_advice) * 100 if number_of_advice > 0 else 0
#         stop_loss_success_rate = (number_of_stop_loss / number_of_advice) * 100 if number_of_advice > 0 else 0
#         take_profit_success_rate = (number_of_take_profit / number_of_advice) * 100 if number_of_advice > 0 else 0
#
#         # Calculate the total success rate (based on profitable actions)
#         total_successful_actions = number_of_buy + number_of_sell + number_of_take_profit
#         total_success_rate = (total_successful_actions / number_of_advice) * 100 if number_of_advice > 0 else 0
#
#         # Call this function after your results are calculated
#         save_results_to_file(scenario, number_of_advice, number_of_buy, buy_success_rate,
#                              number_of_sell, sell_success_rate, number_of_stop_loss,
#                              stop_loss_success_rate, number_of_take_profit,
#                              take_profit_success_rate, total_success_rate)
#
#
# def get_scenarios():
#     all_scenarios = [
#         {
#             'name': "Candle time 1 minute, difference = 0, no EMA/MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 0,
#             'ema': 0,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#         {
#             'name': "Candle time 1 minute, difference = 2, no EMA/MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 2,
#             'ema': 0,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#         {
#             'name': "Candle time 1 minute, difference = 3, no EMA/MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 3,
#             'ema': 0,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#         {
#             'name': "Candle time 1 minute, difference = 0, ema 0.5, no MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 0,
#             'ema': 0.5,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#         {
#             'name': "Candle time 1 minute, difference = 2, ema 0.5, no MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 2,
#             'ema': 0.5,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#         {
#             'name': "Candle time 1 minute, difference = 3, ema 0.5, no MACD, S/P 18/32",
#             'candles': 1,
#             'candle_difference': 3,
#             'ema': 0.5,
#             'macd': 0,
#             'stop_loss_margin': 18,
#             'take_profit_margin': 32
#         },
#     ]
#     return all_scenarios



def main():
    print("Testing a scenario 's")
    fetch_all_candles()
    # all_scenarios = get_scenarios()
    # run_scenario(all_candles)




if __name__ == "__main__":
     main()

"""
    Results scenario: Default
    Number of advices:161
    Number of buy:31
    Succes rate buy: 19.25%
    Number of sell:35
    Success rate sell: 21.74%
    Number of stop loss:27
    Success rate stop loss: 16.77%
    Number of take profit:14
    Succes rate take profit: 8.70%
"""