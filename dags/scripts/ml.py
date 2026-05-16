"""
GAZP ML Pipeline: minio.moex_data → minio.feature_data_gazp_agg
================================================================
Источник:    minio.moex_data.gazp_view
Фичи:        minio.feature_data_gazp_agg.features
Предсказания:minio.feature_data_gazp_agg.predictions
Superset:    minio.feature_data_gazp_agg.v_superset_dashboard

Установка зависимостей:
    pip install trino pandas scikit-learn xgboost lightgbm ta joblib

Запуск:
    python gazp_ml_pipeline_v2.py               # обучение + запись
    python gazp_ml_pipeline_v2.py --mode infer  # только инференс
    python gazp_ml_pipeline_v2.py --csv f.csv   # локальный тест
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
    # ── Укажи реальные значения из своей инфраструктуры ─────────────────
    #
    # КАК НАЙТИ ПРАВИЛЬНЫЙ ХОСТ И ПОРТ:
    #   docker-compose.yml  → сервис trino → ports: "XXXX:8080"  ← вот порт
    #   kubectl get svc     → ищи trino/coordinator
    #   Airflow Connections → conn_id=trino_default → host + port
    #
    # ЧАСТАЯ ОШИБКА — порт 8080 занят Airflow!
    #   Airflow Web UI  обычно висит на :8080
    #   Trino по умолч. тоже :8080, но в compose его часто меняют
    #   → ответ "Airflow 404" означает: попали на Airflow, не на Trino
    #
    # БЫСТРАЯ ПРОВЕРКА (выполни в терминале):
    #   curl http://<host>:<port>/v1/info
    #   Правильно → {"nodeVersion":{"version":"xxx"},"environment":"...",...}
    #   Неправильно → HTML с "Airflow" или "404"
    #
    # Если Trino в Docker, скрипт на Windows-хосте:
    #   host = "localhost"
    #   port = <host_port>  # левая часть из ports: "8081:8080" → 8081
    #
    # Если скрипт тоже в Docker Compose:
    #   host = "trino"      # имя сервиса в docker-compose.yml
    #   port = 8080         # внутренний порт контейнера
    #
    host        = "localhost",  # ← ЗАМЕНИ на IP/hostname Trino
    port        = 8086,         # ← ЗАМЕНИ на порт Trino (не Airflow!)
    user        = "admin",      # ← ЗАМЕНИ на своего пользователя
    http_scheme = "http",       # "https" если Trino за nginx/TLS
)

# Пути в Trino / MinIO
SRC_TABLE      = "minio.moex_data.gazp_view"
FEATURES_TABLE = "minio.feature_data_gazp_agg.features"
PRED_TABLE     = "minio.feature_data_gazp_agg.predictions"

# Параметры модели
MODEL_VERSION  = "v2.0_xgb_lgbm"
MODEL_DIR      = Path("./models")
TP_PCT         = 0.015     # +1.5% take profit
SL_PCT         = 0.008     # -0.8% stop loss
PROBA_BUY_THR  = 0.55      # порог BUY-сигнала
PROBA_SELL_THR = 0.45      # порог SELL-сигнала

# Признаки для ML
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
# 1. ПОДКЛЮЧЕНИЕ К TRINO
# ──────────────────────────────────────────────────────────────

def diagnose_connection():
    """
    Проверяет подключение к Trino перед запуском пайплайна.
    Запусти отдельно: python gazp_ml_pipeline_v2.py --diagnose
    """
    import urllib.request
    import json

    host = TRINO["host"]
    port = TRINO["port"]
    scheme = TRINO["http_scheme"]
    url = f"{scheme}://{host}:{port}/v1/info"

    print(f"\n  Проверяем подключение: {url}")
    try:
        req = urllib.request.Request(url, headers={"X-Trino-User": TRINO["user"]})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode()
            if "nodeVersion" in body:
                info = json.loads(body)
                print(f"  ✓ Trino доступен!")
                print(f"    Версия:  {info.get('nodeVersion', {}).get('version', '?')}")
                print(f"    Среда:   {info.get('environment', '?')}")
                print(f"    Статус:  {'ЗАПУЩЕН' if info.get('starting') == False else 'стартует...'}")
                return True
            else:
                print(f"  ✗ Ответ не от Trino (получен HTML или другой сервис)")
                print(f"    Первые 200 символов ответа: {body[:200]}")
                print(f"\n  Вероятно, на {host}:{port} работает Airflow или другой сервис.")
                print(f"  Найди правильный порт Trino:")
                print(f"    docker ps | grep trino")
                print(f"    docker-compose.yml → ports секция сервиса trino")
                return False
    except ConnectionRefusedError:
        print(f"  ✗ Соединение отклонено — порт {port} закрыт или сервис не запущен")
        return False
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False


def get_conn():
    return connect(
        host       = TRINO["host"],
        port       = TRINO["port"],
        user       = TRINO["user"],
        catalog    = "minio",
        http_scheme= TRINO["http_scheme"],
    )


def trino_select(sql: str) -> pd.DataFrame:
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=cols)


def trino_execute(sql: str):
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.close()


# ──────────────────────────────────────────────────────────────
# 2. ЗАГРУЗКА ДАННЫХ ИЗ TRINO
# ──────────────────────────────────────────────────────────────

def load_features(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Загружает фичи из Trino.

    Логика:
    1. Пробует читать minio.feature_data_gazp_agg.features
    2. Если таблица содержит < 20 колонок (не пересоздана с фичами)
       — автоматически переключается на SRC_TABLE + локальный расчёт.
    3. Печатает подсказку как пересоздать таблицу правильно.
    """
    where = []
    if start_date:
        where.append(f"trade_date >= DATE '{start_date}'")
    if end_date:
        where.append(f"trade_date <= DATE '{end_date}'")
    where_clause = "WHERE " + " AND ".join(where) if where else ""

    # ── Попытка 1: таблица features с готовыми фичами ─────────
    print(f"  Загружаем фичи из {FEATURES_TABLE}...")
    try:
        sql = f"SELECT * FROM {FEATURES_TABLE} {where_clause} ORDER BY start_time"
        df  = trino_select(sql)
        df["start_time"] = pd.to_datetime(df["start_time"])
        df = df.set_index("start_time").sort_index()
        print(f"  → {len(df):,} строк, {len(df.columns)} колонок")

        # Если колонок меньше 20 — таблица неполная (только RAW)
        if len(df.columns) < 20:
            print(f"\n  ⚠  Таблица содержит только {len(df.columns)} колонок.")
            print(f"     SQL-скрипт (Часть 5) ещё не выполнен полностью.")
            print(f"     Фичи будут досчитаны локально через pandas.\n")
            print(f"  Чтобы исправить — выполни в Trino:")
            print(f"    DROP TABLE IF EXISTS {FEATURES_TABLE};")
            print(f"    -- затем Часть 5 из gazp_features_minio.sql\n")
            raise ValueError("features_incomplete")

        return df

    except Exception as e:
        if "features_incomplete" not in str(e):
            print(f"  ⚠  Ошибка чтения {FEATURES_TABLE}: {e}")
            print(f"     Переключаемся на {SRC_TABLE} + локальный расчёт фичей.\n")

    # ── Фолбэк: читаем RAW-витрину и считаем фичи локально ────
    print(f"  Загружаем сырые данные из {SRC_TABLE}...")
    where_raw = []
    if start_date:
        where_raw.append(f"CAST(start_time AS DATE) >= DATE '{start_date}'")
    if end_date:
        where_raw.append(f"CAST(start_time AS DATE) <= DATE '{end_date}'")
    where_raw_clause = "WHERE " + " AND ".join(where_raw) if where_raw else ""

    sql_raw = f"""
        SELECT start_time, open_price, high_price, low_price,
               close_price, volume, trades
        FROM {SRC_TABLE}
        {where_raw_clause}
        ORDER BY start_time
    """
    df_raw = trino_select(sql_raw)
    df_raw["start_time"] = pd.to_datetime(df_raw["start_time"])

    for col in ["open_price","high_price","low_price","close_price","volume"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    if "trades" in df_raw.columns:
        df_raw["trades"] = pd.to_numeric(df_raw["trades"], errors="coerce").fillna(0).astype(int)

    df_raw = df_raw.drop_duplicates(subset=["start_time"]).sort_values("start_time")
    df_raw = df_raw.set_index("start_time")
    print(f"  → {len(df_raw):,} строк из {SRC_TABLE}")

    print(f"  Считаем фичи локально (pandas + ta)...")
    df_raw = compute_features_local(df_raw)
    print(f"  → фичи готовы: {len(df_raw.columns)} колонок")
    return df_raw


# ──────────────────────────────────────────────────────────────
# 3. ПАРСИНГ ЛОКАЛЬНОГО CSV (российский формат)
# ──────────────────────────────────────────────────────────────

def load_local_csv(path: str) -> pd.DataFrame:
    """
    Парсит CSV из GAZP_view с российским форматом чисел:
    '120,49' → 120.49,  '510 390' → 510390
    """
    def ru_num(x):
        if isinstance(x, str):
            x = x.strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
        try:
            return float(x)
        except Exception:
            return np.nan

    df = pd.read_csv(path, sep="\t")

    for col in ["open_price", "high_price", "low_price", "close_price", "volume"]:
        if col in df.columns:
            df[col] = df[col].apply(ru_num)

    if "trades" in df.columns:
        df["trades"] = df["trades"].apply(
            lambda x: int(str(x).replace("\xa0", "").replace(" ", ""))
            if pd.notna(x) else 0
        )

    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.drop_duplicates(subset=["start_time"]).sort_values("start_time")
    return df.set_index("start_time")


# ──────────────────────────────────────────────────────────────
# 4. РАСЧЁТ ФИЧЕЙ ЛОКАЛЬНО (только для CSV-режима)
# ──────────────────────────────────────────────────────────────

def compute_features_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Аналог SQL-скрипта gazp_features_minio.sql, но на pandas + ta.
    В продакшне фичи считаются в Trino и хранятся в features.
    """
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    c, h, l, o, v = (
        df["close_price"], df["high_price"],
        df["low_price"],   df["open_price"], df["volume"]
    )
    trades = df.get("trades", pd.Series(1, index=df.index))

    # ── Свечные признаки ──────────────────────────────────────
    rng = h - l
    df["candle_range"]      = rng
    df["body_size"]         = (c - o).abs()
    df["body_ratio"]        = df["body_size"]  / (rng + 1e-9)
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

    # ── RSI ───────────────────────────────────────────────────
    df["rsi_7"]  = RSIIndicator(close=c, window=7).rsi()
    df["rsi_14"] = RSIIndicator(close=c, window=14).rsi()
    df["rsi_21"] = RSIIndicator(close=c, window=21).rsi()
    df["rsi_divergence"] = df["rsi_7"] - df["rsi_21"]
    df["rsi_zone_code"]  = np.where(df["rsi_14"] >= 70, 1,
                            np.where(df["rsi_14"] <= 30, -1, 0))
    df["rsi_zone"]       = np.where(df["rsi_14"] >= 70, "overbought",
                            np.where(df["rsi_14"] <= 30, "oversold", "neutral"))

    # ── MACD ──────────────────────────────────────────────────
    macd_obj = MACD(close=c, window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()
    df["macd_pct"]    = df["macd"] / (c + 1e-9) * 100
    df["macd_hist_sign"]     = np.sign(df["macd_hist"])
    df["macd_zero_cross"]    = np.where(
        (df["macd"] > 0) & (df["macd"].shift(1) <= 0),  1,
        np.where((df["macd"] < 0) & (df["macd"].shift(1) >= 0), -1, 0))
    df["macd_signal_cross"]  = np.where(
        (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1)),  1,
        np.where(
            (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1)), -1, 0))
    df["sma_12"] = c.rolling(12).mean()
    df["sma_26"] = c.rolling(26).mean()

    # ── Bollinger + ATR + SMA ─────────────────────────────────
    bb = BollingerBands(close=c, window=20)
    df["sma_5"]        = c.rolling(5).mean()
    df["sma_10"]       = c.rolling(10).mean()
    df["sma_20"]       = c.rolling(20).mean()
    df["std_20"]       = c.rolling(20).std()
    df["bb_upper"]     = bb.bollinger_hband()
    df["bb_lower"]     = bb.bollinger_lband()
    df["bb_width_pct"] = (df["bb_upper"] - df["bb_lower"]) / (df["sma_20"] + 1e-9)
    df["bb_pct_b"]     = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["atr_14"]       = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    df["atr_pct"]      = df["atr_14"] / (c + 1e-9) * 100

    # ── Объём ─────────────────────────────────────────────────
    df["vol_sma_5"]    = v.rolling(5).mean()
    df["vol_sma_20"]   = v.rolling(20).mean()
    df["vol_ratio_20"] = v / (df["vol_sma_20"] + 1e-9)
    df["is_high_volume"] = (v > 2 * df["vol_sma_20"]).astype(int)

    # ── Временные признаки ────────────────────────────────────
    df["hour_of_day"]  = df.index.hour
    df["day_of_week"]  = df.index.dayofweek + 1
    df["is_premarket"] = df["hour_of_day"].between(7,  9).astype(int)
    df["is_morning"]   = df["hour_of_day"].between(10, 12).astype(int)
    df["is_midday"]    = df["hour_of_day"].between(13, 15).astype(int)
    df["is_evening"]   = df["hour_of_day"].between(16, 18).astype(int)

    # ── Целевые переменные ────────────────────────────────────
    df["close_next_1h"]       = c.shift(-1)
    df["close_next_3h"]       = c.shift(-3)
    df["target_dir_1h"]       = (df["close_next_1h"] > c).astype(int)
    df["target_dir_3h"]       = (df["close_next_3h"] > c).astype(int)
    df["target_return_1h_pct"]= (df["close_next_1h"] - c) / (c + 1e-9) * 100
    df["trade_date"]          = df.index.date

    return df


# ──────────────────────────────────────────────────────────────
# 5. ПОДГОТОВКА ДАТАСЕТА
# ──────────────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame):
    """Временной split 80/20 — никакой случайной перемешки."""
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠ Признаки отсутствуют: {missing}")

    df_clean = df[feat_cols + [TARGET_DIR, TARGET_PRICE]] \
                 .replace([np.inf, -np.inf], np.nan).dropna()

    split = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split]
    test  = df_clean.iloc[split:]

    print(f"  Train: {len(train):,} строк  "
          f"[{train.index[0].date()} — {train.index[-1].date()}]")
    print(f"  Test:  {len(test):,}  строк  "
          f"[{test.index[0].date()} — {test.index[-1].date()}]")

    X_tr = train[feat_cols]; y_tr_d = train[TARGET_DIR]; y_tr_p = train[TARGET_PRICE]
    X_te = test[feat_cols];  y_te_d = test[TARGET_DIR];  y_te_p = test[TARGET_PRICE]

    return X_tr, X_te, y_tr_d, y_te_d, y_tr_p, y_te_p, test, feat_cols


# ──────────────────────────────────────────────────────────────
# 6. ОБУЧЕНИЕ
# ──────────────────────────────────────────────────────────────

def train_classifier(X_tr, y_tr):
    from xgboost import XGBClassifier
    print("  XGBoost (классификатор ↑↓)...")
    clf = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def train_regressor(X_tr, y_tr):
    from lightgbm import LGBMRegressor
    print("  LightGBM (регрессор цены)...")
    reg = LGBMRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    reg.fit(X_tr, y_tr)
    return reg


def save_models(clf, reg):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / f"clf_{MODEL_VERSION}.pkl")
    joblib.dump(reg, MODEL_DIR / f"reg_{MODEL_VERSION}.pkl")
    print(f"  Модели → {MODEL_DIR}/")


def load_models():
    clf = joblib.load(MODEL_DIR / f"clf_{MODEL_VERSION}.pkl")
    reg = joblib.load(MODEL_DIR / f"reg_{MODEL_VERSION}.pkl")
    return clf, reg


# ──────────────────────────────────────────────────────────────
# 7. ОЦЕНКА
# ──────────────────────────────────────────────────────────────

def evaluate(clf, reg, X_te, y_te_d, y_te_p):
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  mean_absolute_error, mean_absolute_percentage_error)
    pred_dir   = clf.predict(X_te)
    pred_proba = clf.predict_proba(X_te)[:, 1]
    pred_price = reg.predict(X_te)

    print(f"\n{'─'*48}")
    print(f"  Классификатор  Accuracy: {accuracy_score(y_te_d, pred_dir):.4f}"
          f"  F1: {f1_score(y_te_d, pred_dir):.4f}"
          f"  AUC: {roc_auc_score(y_te_d, pred_proba):.4f}")
    print(f"  Регрессор      MAE: {mean_absolute_error(y_te_p, pred_price):.4f} руб"
          f"  MAPE: {mean_absolute_percentage_error(y_te_p, pred_price)*100:.2f}%")
    print(f"{'─'*48}")
    return pred_dir, pred_proba, pred_price


def feature_importance(clf, feat_cols, top_n=15):
    imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n  Топ-{top_n} признаков:")
    print(imp.head(top_n).to_string())
    return imp


# ──────────────────────────────────────────────────────────────
# 8. ЗАПИСЬ ПРЕДСКАЗАНИЙ В TRINO → minio.feature_data_gazp_agg
# ──────────────────────────────────────────────────────────────

def write_predictions(test_df, pred_dir, pred_proba, pred_price, batch_size=500):
    """
    Записывает предсказания в minio.feature_data_gazp_agg.predictions.

    Порядок колонок в INSERT строго соответствует DDL таблицы:
        start_time, close_price, pred_dir_1h, pred_close_1h,
        pred_proba_up, tp_level, sl_level, pred_return_pct,
        signal, model_version, created_at, pred_date   ← партиция последней
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n   = min(len(test_df), len(pred_dir))

    # Trino Hive не поддерживает DELETE на Parquet-партициях.
    # Вместо этого — пропускаем дубликаты через уникальность start_time,
    # либо пересоздаём таблицу перед полной записью (выбирай по задаче).

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

        # Порядок строго по DDL: не-партиционные колонки, pred_date последней
        rows.append(
            f"(TIMESTAMP '{ts}', {close}, {direction}, "
            f"{pred_c:.4f}, {proba_up:.6f}, {tp}, {sl}, {ret_pct}, "
            f"'{signal}', '{MODEL_VERSION}', TIMESTAMP '{now}', DATE '{dt}')"
        )

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        insert_sql = f"""
            INSERT INTO {PRED_TABLE}
                (start_time, close_price, pred_dir_1h, pred_close_1h,
                 pred_proba_up, tp_level, sl_level, pred_return_pct,
                 signal, model_version, created_at,
                 pred_date)
            VALUES {','.join(batch)}
        """
        trino_execute(insert_sql)
        total += len(batch)
        print(f"    → {total}/{len(rows)}")

    print(f"  ✓ {len(rows)} предсказаний записаны в {PRED_TABLE}")


# ──────────────────────────────────────────────────────────────
# 9. SUPERSET-ПОДСКАЗКА
# ──────────────────────────────────────────────────────────────

SUPERSET_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║              ДАШБОРД В APACHE SUPERSET                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ДАТАСЕТ:                                                    ║
║    Database : Trino                                          ║
║    Schema   : feature_data_gazp_agg   (catalog: minio)      ║
║    Table    : v_superset_dashboard                           ║
║                                                              ║
║  ГРАФИКИ:                                                    ║
║  1. Свечной (ECharts Candlestick)                            ║
║     x: start_time                                            ║
║     Open/High/Low/Close: соответствующие поля               ║
║     Доп. серии Line: sma_5, sma_10, sma_20, bb_upper/lower  ║
║                                                              ║
║  2. RSI (Line Chart)                                         ║
║     Metrics: rsi_7, rsi_14, rsi_21                          ║
║     Annotations: горизонтальные линии на 30 и 70            ║
║                                                              ║
║  3. MACD (Mixed Chart = Bar + Line)                          ║
║     Bar:  macd_hist  (цвет через Conditional Formatting)    ║
║     Line: macd, macd_signal                                  ║
║                                                              ║
║  4. Объём (Bar Chart)                                        ║
║     Metric: volume, Цвет: candle_color                       ║
║                                                              ║
║  5. Тепловая карта признаков (Heatmap)                       ║
║     x: trade_date, y: признак, value: значение              ║
║     Признаки: body_ratio, rsi_14, macd_hist,                ║
║               bb_pct_b, atr_pct, vol_ratio_20               ║
║                                                              ║
║  6. Сигналы модели (Table)                                   ║
║     Columns: start_time, close_price, signal,               ║
║              pred_proba_up, tp_level, sl_level              ║
║     Filter: signal IN ('BUY','SELL')                        ║
║                                                              ║
║  ВРЕМЕННОЙ ФИЛЬТР:                                           ║
║     Dashboard → Add Filter → Time Range → start_time        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


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
    pred_dir, pred_proba, pred_price = evaluate(clf, reg, X_te, y_te_d, y_te_p)
    feature_importance(clf, feat_cols)

    if not csv_path:
        print(f"\n[6] Запись предсказаний в Trino...")
        write_predictions(test_df, pred_dir, pred_proba, pred_price)
    else:
        out = test_df[["close_price", "close_next_1h", "target_dir_1h"]].copy()
        out["pred_dir_1h"]   = pred_dir
        out["pred_proba_up"] = pred_proba
        out["pred_close_1h"] = pred_price
        out["signal"]        = np.where(out["pred_proba_up"] >= PROBA_BUY_THR, "BUY",
                               np.where(out["pred_proba_up"] <= PROBA_SELL_THR, "SELL", "HOLD"))
        out.to_csv("gazp_predictions_local.csv")
        print("\n  [CSV] Предсказания → gazp_predictions_local.csv")

    print(SUPERSET_GUIDE)
    return clf, reg


def run_infer():
    print("\n" + "="*50 + "\n  РЕЖИМ: ИНФЕРЕНС\n" + "="*50)
    clf, reg = load_models()
    df = load_features()

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    df_clean  = df[feat_cols + [TARGET_DIR, TARGET_PRICE]] \
                   .replace([np.inf, -np.inf], np.nan).dropna()

    X          = df_clean[feat_cols]
    pred_dir   = clf.predict(X)
    pred_proba = clf.predict_proba(X)[:, 1]
    pred_price = reg.predict(X)

    write_predictions(df_clean, pred_dir, pred_proba, pred_price)
    print("  ✓ Инференс завершён")
    return clf, reg


# ──────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAZP ML Pipeline (minio)")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--diagnose", action="store_true", help="Проверить подключение к Trino")
    parser.add_argument("--csv", type=str, default=None,
                        help="Путь к локальному CSV (тест без Trino)")
    args = parser.parse_args()

    if args.diagnose:
        ok = diagnose_connection()
        if not ok:
            print("\n  Исправь TRINO['host'] и TRINO['port'] в начале скрипта.")
    elif args.mode == "infer":
        run_infer()
    else:
        run_train(csv_path=args.csv)