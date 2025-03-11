import os
import pandas as pd
import requests
from datetime import datetime
import ccxt

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data. For crypto tickers (with a slash), use CCXT (Binance mainnet for public endpoints)."""
    if "/" in ticker:
        # Use a Binance instance WITHOUT API keys for public data
        import ccxt
        binance_public = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        # Do NOT override the API URL; we use mainnet for public endpoints.
        from datetime import datetime
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        try:
            ohlcv = binance_public.fetch_ohlcv(ticker, timeframe='1d', since=since)
        except Exception as e:
            raise Exception(f"Error fetching OHLCV data for {ticker}: {e}")
        
        prices = []
        for entry in ohlcv:
            timestamp, open_, high, low, close, volume = entry
            date_str = datetime.utcfromtimestamp(timestamp/1000).strftime("%Y-%m-%d")
            price_obj = Price(
                open=float(open_),
                close=float(close),
                high=float(high),
                low=float(low),
                volume=int(volume),
                time=date_str
            )
            prices.append(price_obj)
        return prices
    else:
        # For traditional tickers, use the financial API as before.
        if cached_data := _cache.get_prices(ticker):
            filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
            if filtered_data:
                return filtered_data

        headers = {}
        api_key = os.environ.get("FINANCIAL_API_KEY")
        if api_key:
            headers["X-API-KEY"] = api_key.strip()
        url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
        if not prices:
            return []
        _cache.set_prices(ticker, [p.model_dump() for p in prices])
        return prices


# For non-price functions, if the ticker is crypto, return empty lists.
def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
    if "/" in ticker:
        return []
    if cached_data := _cache.get_financial_metrics(ticker):
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]
    headers = {}
    api_key = os.environ.get("FINANCIAL_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key.strip()
    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    metrics_response = FinancialMetricsResponse(**response.json())
    financial_metrics = metrics_response.financial_metrics
    if not financial_metrics:
        return []
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics

def search_line_items(ticker: str, line_items: list[str], end_date: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
    if "/" in ticker:
        return []
    headers = {}
    if api_key := os.environ.get("FINANCIAL_API_KEY"):
        headers["X-API-KEY"] = api_key.strip()
    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []
    return search_results[:limit]

def get_insider_trades(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[InsiderTrade]:
    if "/" in ticker:
        return []
    if cached_data := _cache.get_insider_trades(ticker):
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                         if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                         and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data
    headers = {}
    api_key = os.environ.get("FINANCIAL_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key.strip()
    all_trades = []
    current_end_date = end_date
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        if not insider_trades:
            break
        all_trades.extend(insider_trades)
        if not start_date or len(insider_trades) < limit:
            break
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        if current_end_date <= start_date:
            break
    if not all_trades:
        return []
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades

def get_company_news(ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[CompanyNews]:
    if "/" in ticker:
        return []
    if cached_data := _cache.get_company_news(ticker):
        filtered_data = [CompanyNews(**news) for news in cached_data 
                         if (start_date is None or news["date"] >= start_date)
                         and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data
    headers = {}
    api_key = os.environ.get("FINANCIAL_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key.strip()
    all_news = []
    current_end_date = end_date
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        if not company_news:
            break
        all_news.extend(company_news)
        if not start_date or len(company_news) < limit:
            break
        current_end_date = min(news.date for news in company_news).split('T')[0]
        if current_end_date <= start_date:
            break
    if not all_news:
        return []
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news

def get_market_cap(ticker: str, end_date: str) -> float | None:
    if "/" in ticker:
        return None
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap if financial_metrics else None
    return market_cap

def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
