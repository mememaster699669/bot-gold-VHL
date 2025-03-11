import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    progress.update_status("portfolio_management_agent", None, "Analyzing signals")

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_management_agent", None, "Making trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

    progress.update_status("portfolio_management_agent", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    model_name: str,
    model_provider: str,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages([
        (
          "system",
          """You are a portfolio manager making final trading decisions based on multiple tickers.
          
Trading Rules:
- For long positions:
  * Only buy if you have available cash.
  * Only sell if you currently hold long shares of that ticker.
  * Sell quantity must be ≤ current long position shares.
  * Buy quantity must be ≤ max_shares for that ticker.

- For short positions:
  * Only short if you have available margin (50% of position value required).
  * Only cover if you currently have short shares of that ticker.
  * Cover quantity must be ≤ current short position shares.
  * Short quantity must respect margin requirements.

- The max_shares values are pre-calculated to respect position limits.
- Consider both long and short opportunities based on signals.
- Maintain appropriate risk management with both long and short exposure.

Available Actions:
- "buy": Open or add to a long position.
- "sell": Close or reduce a long position.
- "short": Open or add to a short position.
- "cover": Close or reduce a short position.
- "hold": No action.

Inputs:
- signals_by_ticker: A dictionary mapping each ticker to the signals from various analysts.
- max_shares: Maximum shares allowed per ticker.
- portfolio_cash: Current cash available.
- portfolio_positions: Current positions (both long and short).
- current_prices: Current market prices for each ticker.
- margin_requirement: Current margin requirement for short positions.

Your task: Based on these inputs, generate trading decisions for each ticker. For each ticker, if no valid trading opportunity is identified, output "hold" with quantity 0.

Output strictly in JSON format and nothing else. The JSON must have exactly the following structure:

{{
  "decisions": {{
    "TICKER1": {{
      "action": "buy/sell/short/cover/hold",
      "quantity": integer,
      "confidence": float,
      "reasoning": "string"
    }},
    "TICKER2": {{
      ...
    }}
  }}
}}
"""
        ),
        (
          "human",
          """Based on the team's analysis, make your trading decisions for each ticker using the provided inputs.

Here are the inputs:
- signals_by_ticker: {signals_by_ticker}
- Current Prices: {current_prices}
- Maximum Shares Allowed: {max_shares}
- Portfolio Cash: {portfolio_cash}
- Portfolio Positions: {portfolio_positions}
- Margin Requirement: {margin_requirement}

Output strictly in JSON with the following structure:

{{
  "decisions": {{
    "TICKER1": {{
      "action": "buy/sell/short/cover/hold",
      "quantity": integer,
      "confidence": float,
      "reasoning": "string"
    }},
    "TICKER2": {{
      ...
    }}
  }}
}}

Do not include any explanation or extra text outside the JSON object.
"""
        ),
    ])

    prompt = template.invoke({
        "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
        "current_prices": json.dumps(current_prices, indent=2),
        "max_shares": json.dumps(max_shares, indent=2),
        "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
        "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
        "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
    })

    def create_default_portfolio_output():
        # For testing purposes, simulate a 'buy' decision.
        return PortfolioManagerOutput(decisions={
            ticker: PortfolioDecision(
                action="buy", 
                quantity=1, 
                confidence=80.0, 
                reasoning="Simulated decision for testing purposes"
            ) for ticker in tickers
        })

    result = call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=PortfolioManagerOutput, 
        agent_name="portfolio_management_agent", 
        default_factory=create_default_portfolio_output
    )
    if result is None:
        result = create_default_portfolio_output()
    return result
