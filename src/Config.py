# src/config.py
SYMBOL = "QQQ"          # Nasdaq-100 proxy
INTERVAL = "1d"         # change to "1h" for intraday (limited history on free feeds)
START = "2000-01-01"    # change this for time window
END = "2001-01-01"              # None -> today

HORIZON_DAYS = 5        # label horizon for up/down
FLAT_BAND = 0.001       # +/-0.1% treated as "flat"

RANDOM_SEED = 42