"""
This file will contian the flask backend code, this is where:
- set up the flask server
- define the routes
- call stock API to get stock data 
"""
from flask import Flask, jsonify, render_template, request
import requests
import os

from flask_cors import CORS
app = Flask(__name__)
CORS(app)

ALPHA_VANTAGE_API_KEY = "THPJPWZN8X563GEG"
BASE_URL = "https://www.alphavantage.co/query"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summary/<ticker>', methods=['GET'])
def get_summary(ticker):
    try:
        ticker = ticker.upper()
        
        # Call Alpha Vantage's OVERVIEW endpoint
        params = {
            "function": "OVERVIEW",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        # If ticker is invalid, the response will be empty
        if not data or "Name" not in data:
            return jsonify({"error": "Invalid ticker symbol or no data found"}), 400

        summary = {
            "symbol": ticker,
            "name": data.get("Name", "N/A"),
            "sector": data.get("Sector", "N/A"),
            "industry": data.get("Industry", "N/A"),
            "marketCap": data.get("MarketCapitalization", "N/A"),
            "peRatio": data.get("PERatio", "N/A"),
        }

        return render_template("index.html", summary=summary) 

    except Exception as e:
        import logging
        logging.exception("Exception occurred in auto_complete endpoint")
        return jsonify({"error": "Server error occurred. Please try again later."}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "This endpoint does not exist."}), 404


@app.route('/auto_complete', methods=['GET'])
def auto_complete():
    query = request.args.get("query", "").strip()

    if not query:
        return jsonify([])  # or 400 Bad Request if you'd prefer

    # Alpha Vantage API request
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "bestMatches" not in data:
        return jsonify({"error": f"No symbol matches found for query '{query}'"}), 404

    results = [
        {
            "ticker": match.get("1. symbol", ""),
            "name": match.get("2. name", "")
        }
        for match in data.get("bestMatches", [])
    ]

    return jsonify(results)



if __name__ == '__main__':
    app.run(debug=True)



