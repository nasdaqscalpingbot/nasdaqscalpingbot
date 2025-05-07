import pandas as pd
import numpy as np
from sklearn.utils import resample

def preprocess_data(df):
    """Ensure all numerical columns are properly typed and clean data"""
    # Convert OHLC columns to float
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NA values that resulted from conversion
    df = df.dropna()

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

def label_data(df):
    """Label data with BUY/SELL/HOLD signals"""
    # Support/Resistance
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()

    # Initialize labels
    df['Label'] = 'HOLD'  # Default

    # BUY conditions
    buy_mask = (
            (df['Close'] > df['Resistance'].shift(1)) &  # Price breaks above resistance
            (df['MACD'] > df['Signal']) &  # MACD bullish crossover
            (df['MACD'] > 0) &  # MACD above zero line (stronger trend)
            (df['ATR'] > df['ATR'].rolling(20).mean() * 0.5) &  # Volatility above half of 20-period average
            (df['Close'] > df['Open']) &  # Current candle is bullish
            (df['ADX'] > 25)  # Strong trend confirmation
    )
    df.loc[buy_mask, 'Label'] = 'BUY'

    # SELL conditions
    sell_mask = (
            (df['Close'] < df['Support'].shift(1)) &  # Price breaks below support
            (df['MACD'] < df['Signal']) &  # MACD bearish crossover
            (df['MACD'] < 0) &  # MACD below zero line (stronger trend)
            (df['ATR'] > df['ATR'].rolling(20).mean() * 0.5) &  # Volatility above half of 20-period average
            (df['Close'] < df['Open']) &  # Current candle is bearish
            (df['ADX'] > 25)  # Strong trend confirmation
    )
    df.loc[sell_mask, 'Label'] = 'SELL'

    return df

def balance_dataset(df):
    """Balance the dataset with equal BUY/SELL/HOLD samples"""
    # Group by label
    buy_samples = df[df['Label'] == 'BUY']
    sell_samples = df[df['Label'] == 'SELL']
    hold_samples = df[df['Label'] == 'HOLD']

    # Find the smallest group size
    min_samples = min(len(buy_samples), len(sell_samples), len(hold_samples))

    # Sample equally from each group
    balanced_df = pd.concat([
        buy_samples.sample(min_samples, random_state=42),
        sell_samples.sample(min_samples, random_state=42),
        hold_samples.sample(min_samples, random_state=42)
    ])

    return balanced_df.sample(frac=1, random_state=42)  # Shuffle

def augment_numeric_data(df, target_samples_per_class):
    """Safely augment numerical data with noise"""
    augmented_dfs = []

    for label in ['BUY', 'SELL', 'HOLD']:
        class_data = df[df['Label'] == label].copy()
        current_samples = len(class_data)

        if current_samples < target_samples_per_class:
            # Calculate how many samples to generate
            needed = target_samples_per_class - current_samples

            # Select random samples to duplicate
            samples_to_augment = class_data.sample(needed, replace=True, random_state=42)

            # Add Gaussian noise only to numeric columns
            numeric_cols = samples_to_augment.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, 0.01, size=(needed, len(numeric_cols)))

            # Apply noise while keeping original index
            noisy_samples = samples_to_augment.copy()
            noisy_samples[numeric_cols] += noise

            augmented_dfs.append(pd.concat([class_data, noisy_samples]))
        else:
            augmented_dfs.append(class_data.sample(target_samples_per_class, random_state=42))

    return pd.concat(augmented_dfs).sample(frac=1, random_state=42)  # Shuffle

def main():
    # Load and preprocess data
    df = pd.read_csv("../other/nasdaq_backtest.csv")
    df = preprocess_data(df)

    # Calculate indicators
    df = calculate_indicators(df)

    # Label data
    df = label_data(df)

    # Balance dataset
    balanced_df = balance_dataset(df)

    # Augment data (optional)
    final_dataset = augment_numeric_data(balanced_df, target_samples_per_class=10000)

    # Save final dataset
    final_dataset.to_csv("balanced_training_data.csv", index=False)

    print(f"Final dataset counts:\n{final_dataset['Label'].value_counts()}")

if __name__ == "__main__":
    main()