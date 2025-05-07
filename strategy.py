import market
import numpy as np
import pandas as pd
from ai import make_a_prediction  # Import AI decision-making function
from datetime import datetime, time

def create_dataframe(lst_fetched_candles):
    """Zet de lijst met candles om naar een Pandas DataFrame."""
    df = pd.DataFrame(lst_fetched_candles, columns=['Open', 'Close', 'Timestamp', 'High', 'Low'])
    return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index"""
    # Directional Movement
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    # Smoothed DM and TR
    df['PlusDM'] = df['PlusDM'].rolling(window=period).mean()
    df['MinusDM'] = df['MinusDM'].rolling(window=period).mean()
    df['TR'] = df['TR'].rolling(window=period).mean()

    # Directional Indicators
    df['PlusDI'] = 100 * (df['PlusDM'] / df['TR'])
    df['MinusDI'] = 100 * (df['MinusDM'] / df['TR'])

    # DX and ADX
    df['DX'] = 100 * (abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI']))
    df['ADX'] = df['DX'].rolling(window=period).mean()

    return df['ADX']


def calculate_support_resistance(df, window=14):
    """Berekent support- en resistance-niveaus op basis van een rollend window."""
    df['Support'] = df['Low'].rolling(window=window).min()
    df['Resistance'] = df['High'].rolling(window=window).max()
    return df


def calculate_indicators(df):
    """Calculate all technical indicators"""
    # Calculate ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = np.maximum.reduce([high_low, high_close, low_close])
    df['ATR'] = df['TR'].rolling(window=14).mean()
    # Calculate MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Calculate ADX
    df['ADX'] = calculate_adx(df)
    # Support & Resistance
    df = calculate_support_resistance(df)

    return df

def calculate_atr(candles, period=10):
    """Calculate the Average True Range (ATR) for a given period."""
    if len(candles) < period:
        return None  # Not enough data

    tr_values = []
    for i in range(1, len(candles)):  # Start from index 1 since we need previous close
        high, low = candles[i][3], candles[i][4]  # High & Low of the current candle
        prev_close = candles[i - 1][1]  # Close price of the previous candle

        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(true_range)

    return sum(tr_values[-period:]) / period  # Return the ATR value

def macd_check(lst_fetched_candles):
    """MACD-based trading decision."""
    macd, signal = market.calculate_macd_values(lst_fetched_candles)
    if macd > (signal + 2):
        return "BUY", macd, signal
    elif macd < (signal - 2):
        return "SELL", macd, signal
    else:
        return "HOLD", macd, signal


def three_candle_movement(lst_fetched_candles):
    """Three-candle pattern check for trend confirmation."""
    current_candle_open = lst_fetched_candles[0][0]
    previous_candle_open = lst_fetched_candles[1][0]
    oldest_candle_open = lst_fetched_candles[2][0]

    if current_candle_open > (previous_candle_open + 5) and previous_candle_open > (oldest_candle_open + 5):
        return "BUY"
    elif current_candle_open < (previous_candle_open - 5) and previous_candle_open < (oldest_candle_open - 5):
        return "SELL"
    return "HOLD"


def strategycheck(lst_fetched_candles):
    """Combine traditional strategy with AI model predictions."""

    market_condition = market.detect_market_condition(lst_fetched_candles)
    macd_advice, macd, signal = macd_check(lst_fetched_candles)
    three_candle_advice = three_candle_movement(lst_fetched_candles)
    atr = calculate_atr(lst_fetched_candles)

    bot_advice = "HOLD"
    if macd_advice == three_candle_advice and market_condition == "NORMAL":
        bot_advice = macd_advice  # Can be "BUY", "SELL", or "HOLD"

    dataFrame = create_dataframe(lst_fetched_candles)
    dataFrame = calculate_indicators(dataFrame)

    latest_row = dataFrame.iloc[-1]


    input_features = [
        latest_row['Open'], latest_row['Close'], latest_row['High'], latest_row['Low'],
        latest_row['TR'], latest_row['ATR'], latest_row['MACD'], latest_row['Signal'],
        latest_row['UpMove'], latest_row['DownMove'], latest_row['PlusDM'], latest_row['MinusDM'],
        latest_row['PlusDI'], latest_row['MinusDI'], latest_row['DX'], latest_row['ADX'],
        latest_row['Support'], latest_row['Resistance']
    ]


    # AI only makes a decision if the bot has already signaled a trade
    #if bot_advice in ["BUY", "SELL"]:
    current_time = datetime.now().time()
    # if bot_advice != "HOLD" and market_condition == "NORMAL" and (current_time <= time(1, 0) or current_time >= time(10, 0)):
    if bot_advice != "HOLD" and market_condition == "NORMAL":
        ai_decision, confidence = make_a_prediction(input_features, bot_advice)
        # If AI confidence is low, revert to HOLD
        if confidence < 0.7:
            final_decision = "HOLD"
        else:
            final_decision = ai_decision
            print(f"Final decision: {final_decision}")
    else:
        final_decision = "HOLD"  # Default to HOLD when bot doesn't trigger a trade



    return final_decision, macd, signal  # Return final advice along with MACD values