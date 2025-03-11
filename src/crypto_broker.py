# src/crypto_broker.py

import os
import ccxt

# Load your Binance Testnet API credentials from environment variables
binance_api_key = os.environ.get("BINANCE_API_KEY")
binance_api_secret = os.environ.get("BINANCE_API_SECRET")

if not binance_api_key or not binance_api_secret:
    raise ValueError("Binance Testnet API credentials not found in environment variables.")

# Initialize the Binance Testnet client for spot trading
binance = ccxt.binance({
    'apiKey': binance_api_key.strip(),
    'secret': binance_api_secret.strip(),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    },
})

# Override the API base URL to point to the testnet
binance.urls['api'] = 'https://testnet.binance.vision/api'

def get_balance():
    """Fetch the account balance from Binance Testnet."""
    return binance.fetch_balance()

def submit_order(symbol: str, side: str, order_type: str, amount: float, price: float = None):
    """
    Submit an order to Binance Testnet.
    
    - symbol: Trading pair, e.g., "PAXG/USDT"
    - side: "buy" or "sell"
    - order_type: "market" or "limit"
    - amount: Quantity to trade
    - price: Required for limit orders
    """
    try:
        return binance.create_order(symbol, order_type, side, amount, price)
    except Exception as e:
        print(f"Error submitting order: {e}")
        return None
