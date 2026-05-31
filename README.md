# 🧠 Bitcoin Price Prediction — GRU Deep Learning Model

> **Portfolio Project** — Deep Learning appliqué à la prédiction du prix du Bitcoin  
> Modèle champion : **GRU** (Gated Recurrent Unit)

---

## 📊 Résultats

| Modèle | MAE ($) | RMSE ($) | MAPE (%) |
|--------|---------|----------|----------|
| **GRU** 🏆 | **~1 800** | **~2 300** | **~4%** |
| MLP | ~4 100 | ~5 000 | ~10% |
| Transformer | ~4 200 | ~5 400 | ~10% |
| LSTM | ~7 000 | ~8 200 | ~15% |

---

## 🗂️ Structure du projet

```
Project_Bitcoin_DeepLearning_Best_Model_GRU/
├── model/
│   ├── bitcoin_gru_updated.keras   ← modèle entraîné (2014–2026)
│   └── bitcoin_scaler_updated.pkl  ← MinMaxScaler
├── train.py        ← re-entraîner le modèle avec les données récentes
├── predict.py      ← prédire le prochain prix du Bitcoin
├── requirements.txt
└── TP_Deep_Learning_LSTM.ipynb     ← notebook complet du TP
```

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Utilisation

**Re-entraîner le modèle** (télécharge les données à jour automatiquement via Yahoo Finance) :
```bash
python train.py
```

**Prédire le prochain prix** :
```bash
python predict.py
```

---

## 🏗️ Architecture GRU

```
Input (30 jours × 2 features)
  │
  ├─ Feature 0 : close price (USD)
  └─ Feature 1 : Volume USD
  │
GRU(50 neurones, return_sequences=True)
  │
GRU(50 neurones)
  │
Dense(1) → prix prédit (USD)
```

- **Fenêtre** : 30 jours
- **Features** : `close` + `Volume USD`
- **Optimiseur** : Adam
- **Loss** : MSE
- **Callbacks** : EarlyStopping (patience=5) + ReduceLROnPlateau

---

## 📈 Données

- **Source historique** : BTC-Daily.csv (2014–2022)
- **Mise à jour** : Yahoo Finance via `yfinance` (2022→aujourd'hui)
- **Dataset total** : ~4 200 lignes journalières

---

## ⚠️ Disclaimer

Ce projet est à but **éducatif et de portfolio uniquement**.  
Le modèle ne doit pas être utilisé pour prendre des décisions financières réelles.  
La Directional Accuracy (~50%) confirme que la direction hausse/baisse reste imprévisible avec ces seules features.

---

## 👨‍💻 Auteur

**Paul Cioban** — Projet TP Deep Learning LSTM  
École IA Microsoft x Simplon
