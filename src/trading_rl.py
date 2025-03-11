# trading_rl.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime

class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning.
    Observation: [current_price, cash, holdings]
    Actions: 0 = hold, 1 = buy, 2 = sell
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, price_data: pd.DataFrame, initial_cash=100000):
        super(TradingEnv, self).__init__()
        self.price_data = price_data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.current_step = 0
        
        # Define action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        # Observation space: [current_price, cash, holdings]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        current_price = self.price_data.iloc[self.current_step]['close']
        return np.array([current_price, self.cash, self.holdings], dtype=np.float32)

    def step(self, action):
        done = False
        current_price = self.price_data.iloc[self.current_step]['close']
        reward = 0

        # Execute action
        if action == 1 and self.cash >= current_price:  # buy one unit
            self.cash -= current_price
            self.holdings += 1
        elif action == 2 and self.holdings > 0:  # sell one unit
            self.cash += current_price
            self.holdings -= 1

        self.current_step += 1
        if self.current_step >= len(self.price_data):
            done = True
            reward = self.cash + self.holdings * current_price - self.initial_cash
        else:
            next_price = self.price_data.iloc[self.current_step]['close']
            new_portfolio_value = self.cash + self.holdings * next_price
            old_portfolio_value = self.cash + self.holdings * current_price
            reward = new_portfolio_value - old_portfolio_value

        obs = self._get_obs() if not done else np.array([0, 0, 0], dtype=np.float32)
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0
        return self._get_obs()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash}, Holdings: {self.holdings}")

if __name__ == "__main__":
    # Example training loop
    import pandas as pd
    from stable_baselines3 import PPO

    # Generate dummy price data
    dates = pd.date_range(start="2025-03-01", periods=100)
    prices = np.linspace(100, 150, num=100)
    price_data = pd.DataFrame({'close': prices}, index=dates).reset_index(drop=True)

    env = TradingEnv(price_data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
