"""
GAZP ML Pipeline: minio.moex_data → minio.feature_data_gazp_agg
================================================================
Запуск:
    python gazp_ml_v2_final.py               # обучение + запись
    python gazp_ml_v2_final.py --mode infer  # только инференс
    python gazp_ml_v2_final.py --csv f.csv   # локальный тест
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import trino
from trino.dbapi import connect

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────────────

TRINO = dict(
    host        = "algotrading-trino-1",
    port        = 8080,
    user        = "admin",
    http_scheme = "http",
)

SRC_TABLE           = "minio.moex_data.gazp_view"
FEATURES_TABLE      = "minio.feature_data_gazp_agg.features"
PRED_TABLE          = "minio.feature_data_gazp_agg.predictions"
TABLE_FEATURE_IMP   = "minio.feature_data_gazp_agg.feature_importance"
TABLE_MODEL_METRICS = "minio.feature_data_gazp_agg.model_metrics"

MODEL_VERSION  = "v2.0_xgb"
MODEL_DIR      = Path("./models")
TP_PCT         = 0.015
SL_PCT         = 0.008
PROBA_BUY_THR  = 0.55
PROBA_SELL_THR = 0.45

FEATURE_COLS = [
    "candle_range", "body_size", "body_ratio",
    "upper_wick_ratio", "lower_wick_ratio",
    "close_position", "is_bullish", "candle_return_pct",
    "gap_pct", "vol_per_trade",
    "return_1", "return_2", "return_3", "return_5",
    "rsi_7", "rsi_14", "rsi_21", "rsi_divergence", "rsi_zone_code",
    "macd", "macd_signal", "macd_hist", "macd_pct",
    "macd_hist_sign", "macd_zero_cross", "macd_signal_cross",
    "bb_width_pct", "bb_pct_b", "atr_14", "atr_pct",
    "vol_ratio_20", "is_high_volume",
    "sma_5", "sma_10", "sma_20",
    "hour_of_day", "day_of_week",
    "is_premarket", "is_morning", "is_midday", "is_evening",
]

TARGET_DIR   = "target_dir_1h"
TARGET_PRICE = "close_next_1h"


# ──────────────────────────────────────────────────────────────
# 1. TRINO УТИЛИТЫ
# ──────────────────────────────────────────────────────────────

def get_conn():
    return connect(
        host=TRINO["host"], port=TRINO["port"],
        user=TRINO["user"], catalog="minio",
        http_scheme=TRINO["http_scheme"],
    )

def trino_select(sql: str) -> pd.DataFrame:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=cols)

def trino_execute(sql: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(sql)
    conn.close()


# ──────────────────────────────────────────────────────────────
# 2. ЗАГРУЗКА ДАННЫХ
# ──────────────────────────────────────────────────────────────

def load_features(start_date=None, end_date=None) -> pd.DataFrame:
    where = []
    if start_date:
        where.append(f"trade_date >= DATE '{start_date}'")
    if end_date:
        where.append(f"trade_date <= DATE '{end_date}'")
    where_clause = "WHERE " + " AND ".join(where) if where else ""

    print(f"  Загружаем фичи из {FEATURES_TABLE}...")
    try:
        df = trino_select(
            f"SELECT * FROM {FEATURES_TABLE} {where_clause} ORDER BY start_time"
        )
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.set_index("start_time").sort_index()
        print(f"  -> {len(df):,} строк, {len(df.columns)} колонок")
        if len(df.columns) < 20:
            raise ValueError("features_incomplete")
        return df
    except Exception as e:
        if "features_incomplete" not in str(e):
            print(f"  Переключаемся на {SRC_TABLE} + локальный расчёт.\n")

    print(f"  Загружаем сырые данные из {SRC_TABLE}...")
    df_raw = trino_select(
        f"SELECT start_time, open_price, high_price, low_price, "
        f"close_price, volume, trades FROM {SRC_TABLE} ORDER BY start_time"
    )
    df_raw["start_time"] = pd.to_datetime(df_raw["start_time"])
    for col in ["open_price","high_price","low_price","close_price","volume"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    df_raw["trades"] = pd.to_numeric(df_raw["trades"], errors="coerce").fillna(0).astype(int)
    df_raw = (df_raw.drop_duplicates(subset=["start_time"])
              .sort_values("start_time").set_index("start_time"))
    print(f"  -> {len(df_raw):,} строк из {SRC_TABLE}")
    print("  Считаем фичи локально...")
    df_raw = compute_features_local(df_raw)
    print(f"  -> фичи готовы: {len(df_raw.columns)} колонок")
    return df_raw


# ──────────────────────────────────────────────────────────────
# 3. ЛОКАЛЬНЫЙ CSV
# ──────────────────────────────────────────────────────────────

def load_local_csv(path: str) -> pd.DataFrame:
    def ru_num(x):
        if isinstance(x, str):
            x = x.strip().replace("\xa0","").replace(" ","").replace(",",".")
        try:
            return float(x)
        except Exception:
            return np.nan

    df = pd.read_csv(path, sep="\t")
    for col in ["open_price","high_price","low_price","close_price","volume"]:
        if col in df.columns:
            df[col] = df[col].apply(ru_num)
    if "trades" in df.columns:
        df["trades"] = df["trades"].apply(
            lambda x: int(str(x).replace("\xa0","").replace(" ",""))
            if pd.notna(x) else 0
        )
    df["start_time"] = pd.to_datetime(df["start_time"])
    return (df.drop_duplicates(subset=["start_time"])
            .sort_values("start_time").set_index("start_time"))


# ──────────────────────────────────────────────────────────────
# 4. РАСЧЁТ ФИЧЕЙ ЛОКАЛЬНО
# ──────────────────────────────────────────────────────────────

def compute_features_local(df: pd.DataFrame) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands, AverageTrueRange

    c = df["close_price"]
    h = df["high_price"]
    l = df["low_price"]
    o = df["open_price"]
    v = df["volume"]
    trades = df.get("trades", pd.Series(1, index=df.index))

    rng = h - l
    df["candle_range"]      = rng
    df["body_size"]         = (c - o).abs()
    df["body_ratio"]        = df["body_size"] / (rng + 1e-9)
    df["upper_wick_ratio"]  = (h - df[["open_price","close_price"]].max(axis=1)) / (rng + 1e-9)
    df["lower_wick_ratio"]  = (df[["open_price","close_price"]].min(axis=1) - l) / (rng + 1e-9)
    df["close_position"]    = (c - l) / (rng + 1e-9)
    df["is_bullish"]        = (c > o).astype(int)
    df["candle_return_pct"] = (c - o) / (o + 1e-9) * 100
    df["gap_from_prev"]     = o - c.shift(1)
    df["gap_pct"]           = df["gap_from_prev"] / (c.shift(1) + 1e-9) * 100
    df["vol_per_trade"]     = v / (trades.replace(0, np.nan))
    for lag in [1, 2, 3, 5]:
        df[f"return_{lag}"] = c.pct_change(lag)

    df["rsi_7"]          = RSIIndicator(close=c, window=7).rsi()
    df["rsi_14"]         = RSIIndicator(close=c, window=14).rsi()
    df["rsi_21"]         = RSIIndicator(close=c, window=21).rsi()
    df["rsi_divergence"] = df["rsi_7"] - df["rsi_21"]
    df["rsi_zone_code"]  = np.where(df["rsi_14"] >= 70, 1,
                            np.where(df["rsi_14"] <= 30, -1, 0))

    macd_obj = MACD(close=c, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]             = macd_obj.macd()
    df["macd_signal"]      = macd_obj.macd_signal()
    df["macd_hist"]        = macd_obj.macd_diff()
    df["macd_pct"]         = df["macd"] / (c + 1e-9) * 100
    df["macd_hist_sign"]   = np.sign(df["macd_hist"])
    df["macd_zero_cross"]  = np.where(
        (df["macd"] > 0) & (df["macd"].shift(1) <= 0), 1,
        np.where((df["macd"] < 0) & (df["macd"].shift(1) >= 0), -1, 0))
    df["macd_signal_cross"]= np.where(
        (df["macd"] > df["macd_signal"]) &
        (df["macd"].shift(1) <= df["macd_signal"].shift(1)), 1,
        np.where(
            (df["macd"] < df["macd_signal"]) &
            (df["macd"].shift(1) >= df["macd_signal"].shift(1)), -1, 0))

    bb = BollingerBands(close=c, window=20)
    df["sma_5"]         = c.rolling(5).mean()
    df["sma_10"]        = c.rolling(10).mean()
    df["sma_20"]        = c.rolling(20).mean()
    df["bb_upper"]      = bb.bollinger_hband()
    df["bb_lower"]      = bb.bollinger_lband()
    df["bb_width_pct"]  = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + 1e-9)
    df["bb_pct_b"]      = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["atr_14"]        = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    df["atr_pct"]       = df["atr_14"] / (c + 1e-9) * 100
    df["vol_sma_20"]    = v.rolling(20).mean()
    df["vol_ratio_20"]  = v / (df["vol_sma_20"] + 1e-9)
    df["is_high_volume"]= (v > 2 * df["vol_sma_20"]).astype(int)

    df["hour_of_day"]  = df.index.hour
    df["day_of_week"]  = df.index.dayofweek + 1
    df["is_premarket"] = df["hour_of_day"].between(7,  9).astype(int)
    df["is_morning"]   = df["hour_of_day"].between(10, 12).astype(int)
    df["is_midday"]    = df["hour_of_day"].between(13, 15).astype(int)
    df["is_evening"]   = df["hour_of_day"].between(16, 18).astype(int)

    df["close_next_1h"]        = c.shift(-1)
    df["close_next_3h"]        = c.shift(-3)
    df["target_dir_1h"]        = (df["close_next_1h"] > c).astype(int)
    df["target_dir_3h"]        = (df["close_next_3h"] > c).astype(int)
    df["target_return_1h_pct"] = (df["close_next_1h"] - c) / (c + 1e-9) * 100
    df["trade_date"]           = df.index.date
    return df


# ──────────────────────────────────────────────────────────────
# 5. ПОДГОТОВКА ДАТАСЕТА
# ──────────────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame):
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Признаки отсутствуют: {missing}")

    df_clean = (df[feat_cols + [TARGET_DIR, TARGET_PRICE]]
                .replace([np.inf, -np.inf], np.nan).dropna())
    split = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split]
    test  = df_clean.iloc[split:]

    print(f"  Train: {len(train):,} строк  [{train.index[0].date()} - {train.index[-1].date()}]")
    print(f"  Test:  {len(test):,}  строк  [{test.index[0].date()} - {test.index[-1].date()}]")

    X_tr = train[feat_cols]; y_tr_d = train[TARGET_DIR]; y_tr_p = train[TARGET_PRICE]
    X_te = test[feat_cols];  y_te_d = test[TARGET_DIR];  y_te_p = test[TARGET_PRICE]
    return X_tr, X_te, y_tr_d, y_te_d, y_tr_p, y_te_p, test, feat_cols


# ──────────────────────────────────────────────────────────────
# 6. ОБУЧЕНИЕ
# ──────────────────────────────────────────────────────────────

def train_classifier(X_tr, y_tr):
    from xgboost import XGBClassifier
    print("  XGBoost (классификатор)...")
    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_regressor(X_tr, y_tr):
    from xgboost import XGBRegressor
    print("  XGBoost (регрессор цены)...")
    reg = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        random_state=42, n_jobs=-1,
    )
    reg.fit(X_tr, y_tr)
    return reg


def save_models(clf, reg):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / f"clf_{MODEL_VERSION}.pkl")
    joblib.dump(reg, MODEL_DIR / f"reg_{MODEL_VERSION}.pkl")
    print(f"  Модели -> {MODEL_DIR}/")


def load_models():
    clf = joblib.load(MODEL_DIR / f"clf_{MODEL_VERSION}.pkl")
    reg = joblib.load(MODEL_DIR / f"reg_{MODEL_VERSION}.pkl")
    return clf, reg


# ──────────────────────────────────────────────────────────────
# 7. ОЦЕНКА — возвращает метрики и важность для записи в Trino
# ──────────────────────────────────────────────────────────────

def evaluate(clf, reg, X_te, y_te_d, y_te_p):
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  mean_absolute_error, mean_absolute_percentage_error)
    pred_dir   = clf.predict(X_te)
    pred_proba = clf.predict_proba(X_te)[:, 1]
    pred_price = reg.predict(X_te)

    acc  = accuracy_score(y_te_d, pred_dir)
    f1   = f1_score(y_te_d, pred_dir)
    auc  = roc_auc_score(y_te_d, pred_proba)
    mae  = mean_absolute_error(y_te_p, pred_price)
    mape = mean_absolute_percentage_error(y_te_p, pred_price) * 100

    print(f"\n{'='*48}")
    print(f"  Классификатор  Accuracy: {acc:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
    print(f"  Регрессор      MAE: {mae:.4f} руб  MAPE: {mape:.2f}%")
    print(f"{'='*48}")

    metrics = {
        "accuracy":       round(acc,  6),
        "f1_score":       round(f1,   6),
        "roc_auc":        round(auc,  6),
        "mae":            round(mae,  6),
        "mape":           round(mape, 6),
        "test_rows":      len(X_te),
        "test_date_from": str(y_te_d.index[0].date()),
        "test_date_to":   str(y_te_d.index[-1].date()),
    }
    return pred_dir, pred_proba, pred_price, metrics


def feature_importance(clf, feat_cols, top_n=15):
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Топ-{top_n} признаков:")
    print(imp.head(top_n).to_string())
    return imp


# ──────────────────────────────────────────────────────────────
# 8. ЗАПИСЬ МЕТРИК И ВАЖНОСТИ В TRINO
# ──────────────────────────────────────────────────────────────

def write_feature_importance(imp: pd.Series, ticker: str = "GAZP"):
    """Записывает важность признаков XGBoost в Trino."""
    try:
        trino_execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_FEATURE_IMP} (
                feature_name  VARCHAR,
                importance    DOUBLE,
                rank_num      INTEGER,
                ticker        VARCHAR,
                model_version VARCHAR,
                created_at    TIMESTAMP
            ) WITH (format = 'PARQUET')
        """)
    except Exception:
        pass

    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for rank, (name, val) in enumerate(imp.items()):
        rows.append(
            f"('{name}', {float(val):.8f}, {rank+1}, "
            f"'{ticker}', '{MODEL_VERSION}', TIMESTAMP '{now}')"
        )

    cols = "feature_name, importance, rank_num, ticker, model_version, created_at"
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        trino_execute(
            f"INSERT INTO {TABLE_FEATURE_IMP} ({cols}) VALUES {','.join(batch)}"
        )
    print(f"  -> Важность признаков записана: {len(rows)} строк -> {TABLE_FEATURE_IMP}")


def write_model_metrics(metrics: dict, ticker: str = "GAZP"):
    """Записывает метрики качества модели в Trino."""
    try:
        trino_execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_MODEL_METRICS} (
                ticker         VARCHAR,
                model_version  VARCHAR,
                accuracy       DOUBLE,
                f1_score       DOUBLE,
                roc_auc        DOUBLE,
                mae            DOUBLE,
                mape           DOUBLE,
                test_rows      INTEGER,
                test_date_from VARCHAR,
                test_date_to   VARCHAR,
                created_at     TIMESTAMP
            ) WITH (format = 'PARQUET')
        """)
    except Exception:
        pass

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trino_execute(f"""
        INSERT INTO {TABLE_MODEL_METRICS}
            (ticker, model_version, accuracy, f1_score, roc_auc,
             mae, mape, test_rows, test_date_from, test_date_to, created_at)
        VALUES (
            '{ticker}', '{MODEL_VERSION}',
            {metrics['accuracy']}, {metrics['f1_score']}, {metrics['roc_auc']},
            {metrics['mae']}, {metrics['mape']}, {int(metrics['test_rows'])},
            '{metrics['test_date_from']}', '{metrics['test_date_to']}',
            TIMESTAMP '{now}'
        )
    """)
    print(f"  -> Метрики модели записаны -> {TABLE_MODEL_METRICS}")


# ──────────────────────────────────────────────────────────────
# 9. ЗАПИСЬ ПРЕДСКАЗАНИЙ
# ──────────────────────────────────────────────────────────────

def write_predictions(test_df, pred_dir, pred_proba, pred_price, batch_size=500):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n   = min(len(test_df), len(pred_dir))
    rows = []
    for i in range(n):
        ts       = test_df.index[i].strftime("%Y-%m-%d %H:%M:%S")
        dt       = test_df.index[i].strftime("%Y-%m-%d")
        close    = float(test_df.iloc[i].get("close_price", 0))
        proba_up = float(pred_proba[i])
        pred_c   = float(pred_price[i])
        direction= int(pred_dir[i])
        tp       = round(close * (1 + TP_PCT), 4)
        sl       = round(close * (1 - SL_PCT), 4)
        ret_pct  = round((pred_c - close) / (close + 1e-9) * 100, 4)
        signal   = "BUY"  if proba_up >= PROBA_BUY_THR  else \
                   "SELL" if proba_up <= PROBA_SELL_THR else "HOLD"
        rows.append(
            f"(TIMESTAMP '{ts}', {close}, {direction}, "
            f"{pred_c:.4f}, {proba_up:.6f}, {tp}, {sl}, {ret_pct}, "
            f"'{signal}', '{MODEL_VERSION}', TIMESTAMP '{now}', DATE '{dt}')"
        )

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        trino_execute(f"""
            INSERT INTO {PRED_TABLE}
                (start_time, close_price, pred_dir_1h, pred_close_1h,
                 pred_proba_up, tp_level, sl_level, pred_return_pct,
                 signal, model_version, created_at, pred_date)
            VALUES {','.join(batch)}
        """)
        total += len(batch)
        print(f"    -> {total}/{len(rows)}")
    print(f"  -> {len(rows)} предсказаний записаны в {PRED_TABLE}")


# ──────────────────────────────────────────────────────────────
# 10. ПАЙПЛАЙНЫ
# ──────────────────────────────────────────────────────────────

def run_train(csv_path=None):
    print("\n" + "="*50 + "\n  РЕЖИМ: ОБУЧЕНИЕ\n" + "="*50)

    if csv_path:
        print(f"\n[1] Локальный CSV: {csv_path}")
        df = load_local_csv(csv_path)
        print("[2] Считаем фичи локально...")
        df = compute_features_local(df)
    else:
        print(f"\n[1] Загружаем из Trino: {FEATURES_TABLE}")
        df = load_features()
        print(f"[2] Данные загружены: {len(df):,} строк, {len(df.columns)} колонок.")

    print("\n[3] Подготовка датасета...")
    X_tr, X_te, y_tr_d, y_te_d, y_tr_p, y_te_p, test_df, feat_cols = prepare_dataset(df)

    print("\n[4] Обучение...")
    clf = train_classifier(X_tr, y_tr_d)
    reg = train_regressor(X_tr, y_tr_p)
    save_models(clf, reg)

    print("\n[5] Оценка...")
    pred_dir, pred_proba, pred_price, metrics = evaluate(clf, reg, X_te, y_te_d, y_te_p)
    imp = feature_importance(clf, feat_cols)

    if not csv_path:
        print(f"\n[6] Запись предсказаний в Trino...")
        write_predictions(test_df, pred_dir, pred_proba, pred_price)

        print("\n[7] Запись важности признаков в Trino...")
        write_feature_importance(imp)

        print("\n[8] Запись метрик модели в Trino...")
        try:
            write_model_metrics(metrics)
        except Exception as e:
            print(f"  Метрики не записаны: {e}")

        print(f"\n{'='*50}")
        print("  Все данные записаны в Trino:")
        print(f"    {PRED_TABLE}")
        print(f"    {TABLE_FEATURE_IMP}")
        print(f"    {TABLE_MODEL_METRICS}")
        print(f"{'='*50}")
    else:
        out = test_df[["close_price", "close_next_1h", "target_dir_1h"]].copy()
        out["pred_dir_1h"]   = pred_dir
        out["pred_proba_up"] = pred_proba
        out["pred_close_1h"] = pred_price
        out["signal"] = np.where(out["pred_proba_up"] >= PROBA_BUY_THR, "BUY",
                        np.where(out["pred_proba_up"] <= PROBA_SELL_THR, "SELL", "HOLD"))
        out.to_csv("gazp_predictions_local.csv")
        print("\n  [CSV] Предсказания -> gazp_predictions_local.csv")

    return clf, reg


def run_infer():
    print("\n" + "="*50 + "\n  РЕЖИМ: ИНФЕРЕНС\n" + "="*50)
    clf, reg = load_models()
    df = load_features()
    feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
    df_clean   = (df[feat_cols + [TARGET_DIR, TARGET_PRICE]]
                  .replace([np.inf, -np.inf], np.nan).dropna())
    X          = df_clean[feat_cols]
    pred_dir   = clf.predict(X)
    pred_proba = clf.predict_proba(X)[:, 1]
    pred_price = reg.predict(X)
    write_predictions(df_clean, pred_dir, pred_proba, pred_price)
    print("  Инференс завершён")
    return clf, reg


# ──────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAZP ML Pipeline")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--csv",  type=str, default=None,
                        help="Путь к локальному CSV (без Trino)")
    args = parser.parse_args()

    if args.mode == "infer":
        run_infer()
    else:
        run_train(csv_path=args.csv)