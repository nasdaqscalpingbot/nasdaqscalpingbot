import market


def macd_check(lst_fetched_candles):
    macd, signal = market.calculate_macd_values(lst_fetched_candles)
    if macd > (signal + 2):
        return "BUY", macd, signal
    elif macd < (signal - 2):
        return "SELL", macd, signal
    else:
        return "HOLD", macd, signal

def three_candle_movement(lst_fetched_candles):
    current_candle_open = lst_fetched_candles[0][0]
    previous_candle_open = lst_fetched_candles[1][0]
    oldest_candle_open = lst_fetched_candles[2][0]
    three_candle_advice = "HOLD"
    if current_candle_open > (previous_candle_open + 3) and previous_candle_open > (oldest_candle_open + 3):
        three_candle_advice = "BUY"
    elif current_candle_open < (previous_candle_open - 3) and previous_candle_open < (oldest_candle_open - 3):
        three_candle_advice = "SELL"
    return three_candle_advice


def strategycheck(lst_fetched_candles):
    macd_advice, macd, signal = macd_check(lst_fetched_candles)
    three_candle_advice = three_candle_movement(lst_fetched_candles)
    if macd_advice == three_candle_advice:
        return macd_advice, macd, signal
    else:
        return "HOLD", macd, signal
