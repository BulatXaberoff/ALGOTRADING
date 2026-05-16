"""
Сборщик датасета для ML-модели предсказания акций на Московской бирже.

Источники:
  - MOEX ISS API      → свечи, объём
  - CBR API           → ключевая ставка, курс USD/RUB
  - Investing.com     → цена нефти Brent (через yfinance как BZ=F)
  - Расчётные фичи   → технические индикаторы, лаги, таргет

Установка зависимостей:
  pip install pandas requests yfinance ta tqdm
"""

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from tqdm import tqdm


# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────

TICKER     = "SBER"          # тикер на MOEX (SBER, GAZP, LKOH, ...)
START_DATE = "2020-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
INTERVAL   = 24              # интервал свечей в часах (24 = дневные)
TARGET_DAYS = 1              # горизонт предсказания: через сколько дней


# ─────────────────────────────────────────────
# 1. СВЕЧИ С MOEX ISS API
# ─────────────────────────────────────────────

def fetch_moex_candles(ticker: str, start: str, end: str, interval: int = 24) -> pd.DataFrame:
    """
    Загружает свечи с MOEX ISS API.
    interval: 1=1мин, 10=10мин, 60=1час, 24=1день, 7=1нед, 31=1мес
    """
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/"
        f"{ticker}/candles.json"
    )
    all_rows = []
    start_dt = start
    page = 0

    print(f"Загружаем свечи {ticker} с MOEX...")
    with tqdm() as pbar:
        while True:
            params = {
                "from":     start_dt,
                "till":     end,
                "interval": interval,
                "start":    page * 500,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            candles = data.get("candles", {})
            columns = [c["name"] for c in candles.get("metadata", {}).values()] \
                      if "metadata" in candles else candles.get("columns", [])
            rows    = candles.get("data", [])

            if not rows:
                break

            all_rows.extend(rows)
            pbar.update(len(rows))
            page += 1
            time.sleep(0.2)   # не нагружаем API

    if not all_rows:
        raise ValueError(f"Нет данных для тикера {ticker}. Проверь название.")

    df = pd.DataFrame(all_rows, columns=columns)
    df = df.rename(columns={
        "open":   "open",
        "close":  "close",
        "high":   "high",
        "low":    "low",
        "volume": "volume",
        "begin":  "datetime",
        "end":    "datetime_end",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    print(f"  → {len(df)} свечей загружено")
    return df


# ─────────────────────────────────────────────
# 2. КЛЮЧЕВАЯ СТАВКА ЦБ РФ
# ─────────────────────────────────────────────

def fetch_cbr_key_rate(start: str, end: str) -> pd.Series:
    """
    Загружает историю ключевой ставки ЦБ РФ.
    Возвращает дневной ряд с forward-fill.
    """
    url = "https://www.cbr.ru/hd_base/KeyRate/"
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From":   start,
        "UniDbQuery.To":     end,
    }
    print("Загружаем ключевую ставку ЦБ РФ...")
    try:
        tables = pd.read_html(
            requests.get(url, params=params, timeout=15).text,
            decimal=",", thousands=" "
        )
        df = tables[0]
        df.columns = ["date", "rate"]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df = df.set_index("date").sort_index()
        df["rate"] = df["rate"].astype(float)

        idx = pd.date_range(start, end, freq="D")
        rate = df["rate"].reindex(idx).ffill()
        print(f"  → ключевая ставка: {len(rate)} точек")
        return rate
    except Exception as e:
        print(f"  ! Не удалось загрузить ставку ЦБ: {e}. Пропускаем.")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# 3. КУРС USD/RUB с ЦБ РФ
# ─────────────────────────────────────────────

def fetch_cbr_usd_rub(start: str, end: str) -> pd.Series:
    """
    Загружает курс USD/RUB с сайта ЦБ РФ через XML-API.
    """
    print("Загружаем USD/RUB с ЦБ РФ...")
    try:
        url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
        params = {
            "date_req1": datetime.strptime(start, "%Y-%m-%d").strftime("%d/%m/%Y"),
            "date_req2": datetime.strptime(end,   "%Y-%m-%d").strftime("%d/%m/%Y"),
            "VAL_NM_RQ": "R01235",   # код USD
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.content)
        records = []
        for record in root.findall("Record"):
            date = datetime.strptime(record.attrib["Date"], "%d.%m.%Y")
            value = float(record.find("Value").text.replace(",", "."))
            records.append({"date": date, "usd_rub": value})

        df = pd.DataFrame(records).set_index("date").sort_index()
        idx = pd.date_range(start, end, freq="D")
        series = df["usd_rub"].reindex(idx).ffill()
        print(f"  → USD/RUB: {len(series.dropna())} точек")
        return series
    except Exception as e:
        print(f"  ! Не удалось загрузить USD/RUB: {e}. Пропускаем.")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# 4. ЦЕНА НЕФТИ BRENT (через yfinance)
# ─────────────────────────────────────────────

def fetch_brent(start: str, end: str) -> pd.Series:
    """
    Загружает цену нефти Brent (BZ=F) через yfinance.
    """
    print("Загружаем Brent через yfinance...")
    try:
        brent = yf.download("BZ=F", start=start, end=end, progress=False)
        series = brent["Close"].rename("brent")
        series.index = pd.to_datetime(series.index).tz_localize(None)
        idx = pd.date_range(start, end, freq="D")
        series = series.reindex(idx).ffill()
        print(f"  → Brent: {len(series.dropna())} точек")
        return series
    except Exception as e:
        print(f"  ! Не удалось загрузить Brent: {e}. Пропускаем.")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
# 5. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы на основе OHLCV."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # RSI
    df["rsi_14"] = RSIIndicator(close=c, window=14).rsi()

    # MACD
    macd = MACD(close=c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    # EMA
    df["ema_10"] = EMAIndicator(close=c, window=10).ema_indicator()
    df["ema_30"] = EMAIndicator(close=c, window=30).ema_indicator()
    df["ema_ratio"] = df["ema_10"] / df["ema_30"]

    # Bollinger Bands
    bb = BollingerBands(close=c, window=20)
    df["bb_high"]  = bb.bollinger_hband()
    df["bb_low"]   = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / c
    df["bb_pos"]   = (c - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-9)

    # ATR (волатильность)
    df["atr_14"] = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()

    # Свечные признаки
    df["body_size"]   = (c - df["open"]).abs() / (h - l + 1e-9)
    df["upper_wick"]  = (h - df[["open","close"]].max(axis=1)) / (h - l + 1e-9)
    df["lower_wick"]  = (df[["open","close"]].min(axis=1) - l) / (h - l + 1e-9)
    df["is_bullish"]  = (c > df["open"]).astype(int)

    # Лаги доходности
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag{lag}"] = c.pct_change(lag)

    # Объём
    df["volume_ma5"]   = df["volume"].rolling(5).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma5"] + 1e-9)

    return df


# ─────────────────────────────────────────────
# 6. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ─────────────────────────────────────────────

def add_target(df: pd.DataFrame, days: int = 1) -> pd.DataFrame:
    """
    target_direction: 1 = цена выросла через N дней, 0 = упала/не изменилась
    target_return:    процентное изменение (для регрессии)
    """
    future_close = df["close"].shift(-days)
    df["target_return"]    = (future_close - df["close"]) / df["close"]
    df["target_direction"] = (df["target_return"] > 0).astype(int)
    return df


# ─────────────────────────────────────────────
# 7. СБОРКА ДАТАСЕТА
# ─────────────────────────────────────────────

def build_dataset(
    ticker:      str = TICKER,
    start:       str = START_DATE,
    end:         str = END_DATE,
    interval:    int = INTERVAL,
    target_days: int = TARGET_DAYS,
) -> pd.DataFrame:

    print(f"\n{'='*50}")
    print(f" Сборка датасета: {ticker} | {start} → {end}")
    print(f"{'='*50}\n")

    # --- Свечи ---
    df = fetch_moex_candles(ticker, start, end, interval)

    # --- Технические индикаторы ---
    print("\nСчитаем технические индикаторы...")
    df = add_technical_indicators(df)

    # --- Внешние источники ---
    key_rate = fetch_cbr_key_rate(start, end)
    usd_rub  = fetch_cbr_usd_rub(start, end)
    brent    = fetch_brent(start, end)

    # Выравниваем по торговым дням из df
    dates = df.index.normalize()

    if not key_rate.empty:
        df["key_rate"] = key_rate.reindex(dates).values

    if not usd_rub.empty:
        df["usd_rub"] = usd_rub.reindex(dates).values

    if not brent.empty:
        df["brent"] = brent.reindex(dates).values
        df["brent_return"] = df["brent"].pct_change()

    # --- Целевая переменная ---
    df = add_target(df, target_days)

    # --- Финальная очистка ---
    initial_len = len(df)
    df = df.dropna()
    print(f"\nСтрок после dropna: {len(df)} (удалено {initial_len - len(df)})")

    return df


# ─────────────────────────────────────────────
# 8. ЗАПУСК И СОХРАНЕНИЕ
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset(
        ticker      = TICKER,
        start       = START_DATE,
        end         = END_DATE,
        interval    = INTERVAL,
        target_days = TARGET_DAYS,
    )

    out_path = f"{TICKER}_dataset.csv"
    df.to_csv(out_path)

    print(f"\n{'='*50}")
    print(f" Датасет сохранён: {out_path}")
    print(f" Строк: {len(df)} | Колонок: {len(df.columns)}")
    print(f"\nКолонки:")
    for col in df.columns:
        print(f"  {col}")
    print(f"{'='*50}\n")

    print("Первые 3 строки:")
    print(df.head(3).to_string())