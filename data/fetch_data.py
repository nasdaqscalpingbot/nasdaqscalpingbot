# ============== Imports =====================================

from connection import fetch_us100_snapshot

# ============== Variables =====================================

# ============== Functions =====================================

def fetch_current_market_info():
    us100_snapshot = fetch_us100_snapshot()
    # Extract bid price, netChange, percentageChange, high, and low from the snapshot
    flo_bid = float(us100_snapshot.get('bid', 1))
    flo_close = float(us100_snapshot.get('offer', 1))
    flo_high = float(us100_snapshot.get('high', 1))
    flo_low = float(us100_snapshot.get('low', 1))
    flo_netChange = float(us100_snapshot.get('netChange', 1))
    flo_percentageChange = float(us100_snapshot.get('percentageChange', 1))
    str_time = us100_snapshot.get('updateTime')
    arr_one_candle = [flo_bid, flo_close, flo_high, flo_low, flo_netChange, flo_percentageChange,str_time]
    return arr_one_candle