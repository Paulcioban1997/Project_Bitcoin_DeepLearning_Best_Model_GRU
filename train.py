import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# ─── 1. Télécharger les données Bitcoin via Yahoo Finance ─────────────────────
print("Téléchargement des données BTC-USD...")
btc = yf.download("BTC-USD", start="2014-01-01", interval="1d", progress=False, auto_adjust=True)
btc = btc.reset_index()
btc.columns = [c[0] if isinstance(c, tuple) else c for c in btc.columns]
btc = btc.rename(columns={"Close": "close"})
btc["Date"] = pd.to_datetime(btc["Date"]).dt.normalize()
btc["Volume USD"] = btc["Volume"] * btc["close"]

data = btc[["close", "Volume USD"]].dropna().reset_index(drop=True)
print(f"Données : {len(data)} lignes  ({btc['Date'].min().date()} → {btc['Date'].max().date()})")

# ─── 2. Normalisation ─────────────────────────────────────────────────────────
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ─── 3. Fenêtres glissantes (window = 30 jours) ───────────────────────────────
WINDOW = 30
X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i, :])
    y.append(scaled[i, 0])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")

# ─── 4. Modèle GRU ────────────────────────────────────────────────────────────
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(WINDOW, 2)),
    GRU(50),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

print("\nEntraînement du GRU...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1,
          callbacks=callbacks, verbose=1)

# ─── 5. Évaluation ────────────────────────────────────────────────────────────
preds_raw = model.predict(X_test)
dummy = np.zeros((len(preds_raw), 2))
dummy[:, 0] = preds_raw[:, 0]
predictions = scaler.inverse_transform(dummy)[:, 0]

dummy_r = np.zeros((len(y_test), 2))
dummy_r[:, 0] = y_test
real = scaler.inverse_transform(dummy_r)[:, 0]

mae  = mean_absolute_error(real, predictions)
rmse = np.sqrt(mean_squared_error(real, predictions))
mape = np.mean(np.abs((real - predictions) / real)) * 100

print(f"\nGRU — MAE: {mae:,.0f} $  |  RMSE: {rmse:,.0f} $  |  MAPE: {mape:.1f}%")

# ─── 6. Sauvegarde ────────────────────────────────────────────────────────────
model.save("model/bitcoin_gru_updated.keras")
joblib.dump(scaler, "model/bitcoin_scaler_updated.pkl")
print("\nModèle sauvegardé : model/bitcoin_gru_updated.keras")
print("Scaler sauvegardé : model/bitcoin_scaler_updated.pkl")
