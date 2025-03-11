import sys
import os
import time
import argparse
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from utils.display import print_trading_output
from utils.progress import progress

# Load environment variables
load_dotenv(find_dotenv())

# Import crypto broker functions (for live order execution)
from crypto_broker import submit_order, get_balance

# Import additional decision modules
import trading_rl         # Assumes you have a TradingEnv and RL training/prediction logic
import trading_evolution  # Assumes a main() that returns optimized parameters or a decision

# --- Configuration ---
# For crypto trading, we use Binance Testnet trading pair for tokenized gold (e.g., "PAXG/USDT")
TICKERS = ["PAXG/USDT"]

# Default portfolio for simulation/reference
PORTFOLIO = {
    "cash": 100000.0,
    "positions": {ticker: {"long": 0, "short": 0} for ticker in TICKERS},
}

# Default model and provider for LLM-based decisions (if used)
DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_MODEL_PROVIDER = "OpenAI"

# For LLM, we use your existing technical analysis-based agents
DEFAULT_ANALYSTS = ["technical_analyst"]

# Polling interval in seconds
POLLING_INTERVAL = 60

# --- Helper Functions ---

def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        import json
        return json.loads(response)
    except Exception as e:
        print(f"Error parsing response: {e}\nResponse: {repr(response)}")
        return None

def submit_order_live(decision: dict):
    """
    Submit a live order via our crypto broker.
    Expects decision as a dict with keys:
      - asset (trading pair; default: TICKERS[0])
      - action: "buy" or "sell"
      - quantity: number of units to trade
    """
    if not isinstance(decision, dict):
        print(f"Skipping order submission; decision is not structured: {decision}")
        return None
    asset = decision.get("asset", TICKERS[0])
    action = decision.get("action")
    quantity = decision.get("quantity", 0)
    if not asset or not action or quantity <= 0:
        print(f"Invalid decision format: {decision}")
        return None
    print(f"Submitting live order: {action.upper()} {quantity} units of {asset}")
    order = submit_order(asset, action, "market", quantity)
    return order

def get_llm_decision(tickers, portfolio, model_name, model_provider):
    """
    Run one cycle of your LLM-based decision using your existing workflow.
    Returns a dictionary mapping each ticker to its decision.
    """
    from agents.technicals import technical_analyst_agent
    from agents.risk_manager import risk_management_agent
    from agents.portfolio_manager import portfolio_management_agent
    from utils.analysts import get_analyst_nodes
    from graph.state import AgentState
    import json

    # Build a simple workflow with your LLM-based agents:
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", lambda state: state)
    analyst_nodes = get_analyst_nodes()
    for analyst_key in DEFAULT_ANALYSTS:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    for analyst_key in DEFAULT_ANALYSTS:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    workflow.set_entry_point("start_node")
    app = workflow.compile()

    now = datetime.now()
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(minutes=1)).strftime("%Y-%m-%d")
    final_state = app.invoke({
        "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": model_name,
            "model_provider": model_provider,
        },
    })
    try:
        decisions = json.loads(final_state["messages"][-1].content)
    except Exception as e:
        print("LLM decision parsing error:", e)
        decisions = {ticker: {"action": "hold", "quantity": 0, "confidence": 0.0, "reasoning": "LLM fallback"} for ticker in tickers}
    return decisions

def get_rl_decision(tickers, portfolio):
    """
    Run one cycle using the pre-trained RL agent.
    Returns a decision dictionary mapping ticker to decision.
    """
    from stable_baselines3 import PPO
    from trading_rl import TradingEnv
    from api import get_prices, prices_to_df
    now = datetime.now()
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
    price_data = get_prices(tickers[0], start_date, end_date)
    if not price_data:
        print("RL: No price data available.")
        return {tickers[0]: {"action": "hold", "quantity": 0, "confidence": 0.0, "reasoning": "No data"}}
    price_df = prices_to_df(price_data)
    env = TradingEnv(price_df, initial_cash=portfolio.get("cash", 100000.0))
    try:
        model = PPO.load("src/rl_model.zip", env=env)
    except Exception as e:
        print("RL model load error:", e)
        return {tickers[0]: {"action": "hold", "quantity": 0, "confidence": 0.0, "reasoning": "RL model error"}}
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        # Convert the action from a numpy array to an integer
        action = int(action)
        obs, reward, done, _ = env.step(action)
    action_meaning = {0: "hold", 1: "buy", 2: "sell"}
    decision = {tickers[0]: {"action": action_meaning.get(action, "hold"), "quantity": 1, "confidence": 80.0, "reasoning": "RL decision"}}
    return decision

def get_evo_decision(tickers, portfolio):
    """
    Run the evolutionary algorithm to optimize parameters.
    Returns a decision dictionary mapping ticker to decision.
    """
    import trading_evolution
    best_params = trading_evolution.main()  # Adjust trading_evolution.main() to return best parameters
    decision = {tickers[0]: {"action": "buy", "quantity": 1, "confidence": 75.0, "reasoning": f"Optimized params: {best_params}"}}
    return decision

def combine_decisions(decisions_list):
    """
    Combine decisions from LLM, RL, and Evo approaches.
    Uses a simple majority vote for the action and averages the quantities and confidence.
    """
    from collections import Counter
    actions = []
    quantities = []
    confidences = []
    reasonings = []
    for d in decisions_list:
        # Assume each d is a dict with a single key (the ticker)
        for ticker, decision in d.items():
            actions.append(decision.get("action", "hold"))
            quantities.append(decision.get("quantity", 0))
            confidences.append(decision.get("confidence", 0))
            reasonings.append(decision.get("reasoning", ""))
    action_counts = Counter(actions)
    final_action = action_counts.most_common(1)[0][0]
    final_quantity = int(round(sum(quantities) / len(quantities))) if quantities else 0
    final_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    final_reasoning = "; ".join(reasonings)
    return {"asset": TICKERS[0], "action": final_action, "quantity": final_quantity, "confidence": final_confidence, "reasoning": final_reasoning}

def run_live_trading(tickers, portfolio, model_name, model_provider, polling_interval=POLLING_INTERVAL):
    """
    Runs the trading bot in a live loop using all three decision approaches concurrently.
    Their decisions are combined into one final decision.
    """
    print("Starting live trading for:", tickers)
    while True:
        now = datetime.now()
        # For LLM and RL cycles, we use a 1-minute window for LLM and 5 days for RL data.
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(minutes=1)).strftime("%Y-%m-%d")
        print(f"\nPolling market data from {start_date} to {end_date}...")

        # Get decisions from each approach
        llm_dec = get_llm_decision(tickers, portfolio, model_name, model_provider)
        rl_dec = get_rl_decision(tickers, portfolio)
        evo_dec = get_evo_decision(tickers, portfolio)

        decisions_list = [llm_dec, rl_dec, evo_dec]
        final_decision = combine_decisions(decisions_list)
        print("Combined Decision:", final_decision)

        print_trading_output({"decisions": {TICKERS[0]: final_decision}, "analyst_signals": {}})

        # Submit the final combined order:
        submit_order_live(final_decision)

        print(f"Waiting for {polling_interval} seconds until next cycle...")
        time.sleep(polling_interval)

def submit_order_live(decision: dict):
    """
    Submit a live order using the crypto broker.
    """
    if not isinstance(decision, dict):
        print(f"Skipping order submission; decision is not structured: {decision}")
        return None
    asset = decision.get("asset", TICKERS[0])
    action = decision.get("action")
    quantity = decision.get("quantity", 0)
    
    if not asset or not action or quantity <= 0:
        print(f"Invalid decision format: {decision}")
        return None

    print(f"Submitting live order: {action.upper()} {quantity} units of {asset}")
    order = submit_order(asset, action, "market", quantity)
    return order

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot using combined RL, Evo, and LLM approaches concurrently")
    parser.add_argument("--polling", type=int, default=POLLING_INTERVAL, help="Polling interval in seconds")
    args = parser.parse_args()
    polling_interval = args.polling

    try:
        run_live_trading(TICKERS, PORTFOLIO, DEFAULT_MODEL_NAME, DEFAULT_MODEL_PROVIDER, polling_interval)
    except KeyboardInterrupt:
        print("\nLive trading interrupted. Exiting...")

# poetry run python src/main.py --polling 60
