from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
import numpy as np
import joblib
import requests
from keras.models import load_model

app = FastAPI()

# ✅ CORS settings for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "AI Prediction API Running"}

# ✅ 1. AI Price Direction Prediction
@app.get("/predict")
def predict(coin: str):
    try:
        with open(f"data/{coin}.json", "r") as f:
            prices = json.load(f)

        if len(prices) < 60:
            return {"error": "Not enough data. Minimum 60 prices required."}

        last_60 = [entry["price"] for entry in prices[-60:]]
        last_60 = np.array(last_60).reshape(-1, 1)

        scaler = joblib.load(f"models/{coin}_scaler.gz")
        model = load_model(f"models/{coin}.h5")

        scaled = scaler.transform(last_60)
        X = np.array(scaled).reshape(1, 60, 1)

        pred = model.predict(X)
        if len(pred.shape) == 2 and pred.shape[1] == 1:
            predicted_price_scaled = pred[0][0]
        else:
            return {"error": f"Unexpected prediction shape: {pred.shape}"}

        predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]
        last_price = last_60[-1][0]

        direction = "up" if predicted_price > last_price else "down"
        confidence = abs(predicted_price - last_price) / last_price

        return {
            "prediction": direction,
            "confidence": round(float(confidence), 4),
            "predicted_price": round(float(predicted_price), 2),
            "last_price": round(float(last_price), 2)
        }

    except Exception as e:
        return {"error": str(e)}

# ✅ 2. List of Available Coins
@app.get("/available-coins")
def available_coins():
    try:
        files = os.listdir("data")
        coins = [f.replace(".json", "") for f in files if f.endswith(".json")]
        return {"coins": coins}
    except Exception as e:
        return {"error": str(e)}

# ✅ 3. Debug: Show last 60 prices
@app.get("/debug-prices")
def debug_prices(coin: str):
    try:
        with open(f"data/{coin}.json", "r") as f:
            prices = json.load(f)
        cleaned = [entry["price"] if isinstance(entry, dict) and "price" in entry else entry for entry in prices]
        return {"coin": coin, "last_60_prices": cleaned[-60:]}
    except Exception as e:
        return {"error": str(e)}

# ✅ 4. Proxy Coin Details (for coin.html)
@app.get("/proxy/coin/{coin}")
def get_coin_data(coin: str):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}?localization=false&tickers=false&market_data=true"
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"Failed to fetch coin data from CoinGecko for '{coin}'"}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ✅ 5. Proxy: Top 100 market data
@app.get("/proxy/markets")
def get_top_coins():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if not isinstance(data, list):
            raise ValueError("Invalid response format from CoinGecko")

        print(f"[INFO] ✅ Loaded {len(data)} coins from CoinGecko")
        return data

    except Exception as e:
        print(f"[WARN] ❌ CoinGecko failed. Loading local fallback: {e}")
        try:
            with open("data/markets.json", "r") as f:
                fallback_data = json.load(f)
            return fallback_data
        except Exception as fallback_err:
            print(f"[ERROR] ❌ Fallback load failed: {fallback_err}")
            return JSONResponse(status_code=500, content={"error": str(fallback_err)})


# ✅ 6. AI Signal Endpoint (buy/sell/hold)
@app.get("/signal")
def get_signal(coin: str):
    try:
        with open(f"data/{coin}.json", "r") as f:
            prices = json.load(f)

        if len(prices) < 2:
            return {"error": "Not enough data"}

        last_price = prices[-1]["price"]
        prev_price = prices[-2]["price"]
        change = last_price - prev_price

        if abs(change) < 1:
            signal = "hold"
        elif change > 0:
            signal = "buy"
        else:
            signal = "sell"

        return {"signal": signal, "change": round(change, 4)}

    except Exception as e:
        return {"error": str(e)}

# ✅ 7. Local Coin Data (for AI + chart integration)
@app.get("/local-coin-data")
def local_coin_data(coin: str):
    try:
        with open(f"data/{coin}.json", "r") as f:
            prices = json.load(f)
        if not prices:
            return {"error": "No price data"}
        last = prices[-1]["price"]
        return {
            "name": coin.capitalize(),
            "symbol": coin[:3].upper(),
            "price": last,
            "market_cap": 0,
            "volume": 0,
            "circulating_supply": 0,
            "total_supply": 0,
            "homepage": "https://example.com"
        }
    except Exception as e:
        return {"error": str(e)}
@app.get("/proxy/markets")
def get_top_coins():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1
        }
        response = requests.get(url, params=params)
        data = response.json()

        # 🔒 Ensure API response is a list
        if not isinstance(data, list):
            raise ValueError("Invalid response format from CoinGecko")

        print(f"[INFO] ✅ Loaded {len(data)} coins from CoinGecko")
        return JSONResponse(content=data)  # ✅ RETURN A LIST DIRECTLY

    except Exception as e:
        print(f"[ERROR] Failed to load top coins: {e}")
        return JSONResponse(content=[])  # ✅ Must return an array always

