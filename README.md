# Bitcoin Forecast AI

Forecasting Bitcoin next-day price using Deep Learning.

---

## Models Tested

| Model | MAPE (%) | MAE ($) | R² |
|-------|----------|---------|-----|
| **GRU + RSI + MACD** 🏆 | **3.05** | **2,703** | **0.968** |
| GRU (baseline) | 3.19 | 2,828 | — |
| GRU Huber loss | 4.49 | 4,140 | 0.924 |
| Bidirectional GRU | 4.88 | 4,455 | 0.918 |
| TiDE | — | — | — |
| TFT | — | — | — |
| PatchTST | — | — | — |
| N-BEATS | — | — | — |
| Transformer | — | — | — |
| GRU + Attention | — | — | — |
| LSTM | 7.16 | 6,529 | 0.829 |
| MLP | — | — | — |

> All models trained on the same dataset: BTC daily data 2014–2026 (4,189 rows).

---

## Best Model

**GRU + RSI + MACD**

```
Input (30 days x 5 features)
  |
  |- close price (USD)
  |- Volume USD
  |- RSI (14)
  |- MACD
  `- Signal Line
  |
GRU(64 units, return_sequences=True)
  |
Dropout(0.2)
  |
GRU(32 units)
  |
Dropout(0.2)
  |
Dense(1) -> predicted price (USD)
```

---

## Results

```
MAPE:  3.05%
R2:    0.968
MAE:   $2,703
```

---

## Project Structure

```
Project_Bitcoin_DeepLearning_Best_Model_GRU/
|-- model/
|   |-- bitcoin_champion_final.keras     <- trained champion model (2014-2026)
|   |-- bitcoin_scaler_champion.pkl      <- MinMaxScaler (5 features)
|   |-- bitcoin_features_champion.json   <- feature list
|   |-- bitcoin_gru_updated.keras        <- baseline GRU (close + Volume)
|   `-- bitcoin_scaler_updated.pkl       <- baseline scaler
|-- train.py                             <- retrain with latest data (yfinance)
|-- predict.py                           <- predict next BTC price
|-- requirements.txt
`-- TP_Deep_Learning_LSTM.ipynb          <- full research notebook (10 parts)
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Retrain the model** (downloads latest data automatically via Yahoo Finance):
```bash
python train.py
```

**Predict next price**:
```bash
python predict.py
```

---

## Data

- **Source**: Yahoo Finance via `yfinance` (BTC-USD, 2014-2026)
- **Total dataset**: ~4,189 daily rows
- **Train/Test split**: 80/20
- **Window size**: 30 days

---

## Notebook Structure

The notebook (`TP_Deep_Learning_LSTM.ipynb`) covers 10 parts:

1. Data loading & preprocessing
2. Exploratory Data Analysis (EDA)
3. MLP baseline
4. LSTM
5. GRU
6. Bidirectional GRU
7. Transformer
8. Data update (2014-2026 via yfinance)
9. Researcher experiments (7 feature/architecture combos)
10. State-of-the-art comparison (N-BEATS, PatchTST, TiDE, TFT)

---

## Disclaimer

This project is for **educational and portfolio purposes only**.
Do not use this model for real financial decisions.

---

## Author

**Paul Cioban** -- TP Deep Learning
Ecole IA Microsoft x Simplon
