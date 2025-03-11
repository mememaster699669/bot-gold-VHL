# src/train_rl.py
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from trading_rl import TradingEnv

def train_and_save_rl_model():
    # Generate dummy price data
    dates = pd.date_range(start="2025-03-01", periods=100)
    prices = np.linspace(100, 150, num=100)
    price_data = pd.DataFrame({'close': prices}, index=dates).reset_index(drop=True)

    # Create the trading environment
    env = TradingEnv(price_data, initial_cash=100000)
    
    # Train the RL model using PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    
    # Save the trained model to 'src/rl_model.zip'
    model.save("src/rl_model")
    print("RL model has been saved to src/rl_model.zip")

if __name__ == "__main__":
    train_and_save_rl_model()
