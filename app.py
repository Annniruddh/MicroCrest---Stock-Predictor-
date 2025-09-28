# app.py
from flask import Flask, request, jsonify
from predict import load_model_and_scaler, predict_next_day
import traceback

app = Flask(__name__)

# Simple in-memory cache of loaded models (per-symbol)
MODEL_CACHE = {}

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    symbol = request.args.get("symbol", default=None, type=str)
    if not symbol:
        return jsonify({"error": "Provide `symbol` query param, e.g. /predict?symbol=AAPL"}), 400
    symbol = symbol.upper()
    try:
        if symbol not in MODEL_CACHE:
            model, scaler = load_model_and_scaler(symbol)
            MODEL_CACHE[symbol] = (model, scaler)
        else:
            model, scaler = MODEL_CACHE[symbol]
        result = predict_next_day(symbol, model=model, scaler=scaler, fetch_days=120)
        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "trace": tb}), 500

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
