import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf

# ─── Charger le modèle et le scaler ──────────────────────────────────────────
model  = tf.keras.models.load_model("model/bitcoin_gru_updated.keras")
scaler = joblib.load("model/bitcoin_scaler_updated.pkl")

# ─── Récupérer les 30 derniers jours de BTC ──────────────────────────────────
btc = yf.download("BTC-USD", period="60d", interval="1d", progress=False, auto_adjust=True)
btc = btc.reset_index()
btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]
btc = btc.rename(columns={"Close": "close"})
btc["Volume USD"] = btc["Volume"] * btc["close"]

last_30 = btc[["close", "Volume USD"]].tail(30).values   # shape (30, 2)
last_price = float(last_30[-1, 0])

# ─── Normaliser + prédire ─────────────────────────────────────────────────────
scaled = scaler.transform(last_30)
X = scaled.reshape(1, 30, 2)

pred_scaled = model.predict(X, verbose=0)
dummy = np.zeros((1, 2))
dummy[0, 0] = pred_scaled[0, 0]
pred_price = scaler.inverse_transform(dummy)[0, 0]

# ─── Résultat ─────────────────────────────────────────────────────────────────
print(f"Prix actuel (dernier connu)  : ${last_price:,.2f}")
print(f"Prix prédit (prochain jour)  : ${pred_price:,.2f}")
diff = pred_price - last_price
print(f"Variation prédite            : {'+' if diff >= 0 else ''}{diff:,.2f} $ ({diff/last_price*100:+.2f}%)")
