# src/config.py
SYMBOL = "QQQ"          
INTERVAL = "1d"         # change to "1h" for intraday (limited history on free feeds)
START = "2000-01-01"    
END = "2024-01-01"              

HORIZON_DAYS = 5        # label horizon for up/down
FLAT_BAND = 0.001       # +/-0.1% treated as "flat"

RANDOM_SEED = 42
