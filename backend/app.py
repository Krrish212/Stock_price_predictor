import os
import requests
from flask import Flask, render_template, request, jsonify

from backend.models.lstm_model import train_or_load_for_symbol

# ---------- Config ----------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------- Helpers ----------
def fetch_company_overview(ticker: str) -> dict | None:
    """Call Alpha Vantage OVERVIEW endpoint for company fundamentals."""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=20)
    data = resp.json()
    # If the ticker is invalid, Alpha Vantage returns an empty dict or missing keys
    if not data or "Name" not in data:
        return None
    return {
        "symbol": ticker,
        "name": data.get("Name", "N/A"),
        "sector": data.get("Sector", "N/A"),
        "industry": data.get("Industry", "N/A"),
        "marketCap": data.get("MarketCapitalization", "N/A"),
        "peRatio": data.get("PERatio", "N/A"),
        "website": data.get("Website", "N/A"),
    }


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html", summary=None, plots=None, next_day=None, ticker=None)


@app.route("/auto_complete")
def auto_complete():
    query = (request.args.get("query") or "").upper()
    # Placeholder suggestions; plug in a real search later if needed
    sample = [
        {"ticker": "AAPL", "name": "Apple Inc."},
        {"ticker": "MSFT", "name": "Microsoft Corp."},
        {"ticker": "GOOGL", "name": "Alphabet Inc."},
        {"ticker": "AMZN", "name": "Amazon.com Inc."},
        {"ticker": "TSLA", "name": "Tesla Inc."},
    ]
    return jsonify([s for s in sample if s["ticker"].startswith(query)])


# HTML page: show summary + plots + next-day prediction
@app.route("/summary/<symbol>")
def summary(symbol):
    symbol = symbol.upper()

    # Try to fetch fundamentals for the page (if API key set)
    summary_meta = fetch_company_overview(symbol) or {
        "name": symbol,
        "symbol": symbol,
        "sector": "—",
        "industry": "—",
        "marketCap": "—",
        "peRatio": "—",
        "website": "—",
    }

    try:
        # Train/load + predict + save plots; set epochs=0 for inference-only (require pre-trained)
        result = train_or_load_for_symbol(symbol, epochs=50)

        return render_template(
            "index.html",
            summary=summary_meta,
            plots=result["plot_urls"],
            next_day=result["next_price"],
            ticker=symbol,
        )
    except Exception as e:
        # Render page with error message
        return render_template(
            "index.html",
            summary=summary_meta,
            plots=None,
            next_day=None,
            ticker=symbol,
            error=str(e),
        )


# JSON API: just fundamentals (Alpha Vantage OVERVIEW)
@app.route("/api/summary/<ticker>", methods=["GET"])
def get_summary(ticker):
    try:
        ticker = ticker.upper()
        data = fetch_company_overview(ticker)
        if not data:
            return jsonify({"error": "Invalid ticker symbol or no data found"}), 400
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# 404 handler for API-style routes
@app.errorhandler(404)
def page_not_found(e):
    # For browser routes you render templates; this keeps a simple JSON for unmapped endpoints
    return jsonify({"error": "This endpoint does not exist."}), 404


if __name__ == "__main__":
    # Run from project root:
    #   PYTHONPATH=. python -m backend.app
    app.run(debug=True)