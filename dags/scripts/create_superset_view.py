"""
Создание VIEW minio.feature_data_gazp_agg.v_superset_dashboard
==============================================================
VIEW читает напрямую из minio.moex_data.gazp_view — не зависит
от таблицы features. Все индикаторы считаются на лету через CTE.

Запуск:
    python create_superset_view.py
"""

import trino
import os

conn = trino.dbapi.connect(
    host=os.getenv("TRINO_HOST", "algotrading-trino-1"),
    port=int(os.getenv("TRINO_PORT", 8080)),
    user="admin",
    catalog="minio",
    schema="feature_data_gazp_agg",
)
cur = conn.cursor()

CATALOG        = "minio"
FEATURE_SCHEMA = "feature_data_gazp_agg"
RAW_VIEW       = f"{CATALOG}.moex_data.gazp_view"
PRED_TABLE     = f"{CATALOG}.{FEATURE_SCHEMA}.predictions"


def execute(sql: str, message: str = ""):
    if message:
        print(message)
    cur.execute(sql)
    print("    ✓ OK")


def drop_view():
    try:
        cur.execute(f"DROP VIEW IF EXISTS {CATALOG}.{FEATURE_SCHEMA}.v_superset_dashboard")
        print("    ✓ Старый VIEW удалён")
    except Exception as e:
        print(f"    ⚠ {e}")


def create_superset_dashboard_view():
    execute(f"""
CREATE OR REPLACE VIEW {CATALOG}.{FEATURE_SCHEMA}.v_superset_dashboard AS
-- ══════════════════════════════════════════════════════════════
-- Читаем напрямую из RAW-витрины + считаем индикаторы через CTE.
-- Не зависит от таблицы features.
-- Предсказания подключаются через LEFT JOIN — если нет, NULL.
-- ══════════════════════════════════════════════════════════════
WITH

-- ── RSI 7 и 14 ────────────────────────────────────────────────
rsi_calc AS (
    WITH delta AS (
        SELECT
            start_time,
            close_price - LAG(close_price) OVER (ORDER BY start_time) AS d
        FROM {RAW_VIEW}
    ),
    gl AS (
        SELECT
            start_time,
            GREATEST(d,  0) AS gain,
            GREATEST(-d, 0) AS loss
        FROM delta
    )
    SELECT
        start_time,
        -- RSI-14
        CASE
            WHEN AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) = 0
            THEN 100.0
            ELSE ROUND(100.0 - 100.0 / (1.0 +
                 AVG(gain) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) /
                 AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)), 2)
        END AS rsi_14,
        -- RSI-7
        CASE
            WHEN AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) = 0
            THEN 100.0
            ELSE ROUND(100.0 - 100.0 / (1.0 +
                 AVG(gain) OVER (ORDER BY start_time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) /
                 AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)), 2)
        END AS rsi_7
    FROM gl
),

-- ── MACD (12, 26, 9) ──────────────────────────────────────────
macd_calc AS (
    WITH sma AS (
        SELECT
            start_time,
            AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS sma_12,
            AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS sma_26
        FROM {RAW_VIEW}
    ),
    ml AS (
        SELECT
            start_time,
            ROUND(sma_12 - sma_26, 6) AS macd
        FROM sma
    )
    SELECT
        start_time,
        macd,
        ROUND(AVG(macd) OVER (ORDER BY start_time ROWS BETWEEN 8 PRECEDING AND CURRENT ROW), 6) AS macd_signal
    FROM ml
),

-- ── Bollinger Bands + SMA + ATR + Объём ───────────────────────
bb_calc AS (
    WITH tr AS (
        SELECT
            start_time,
            close_price,
            high_price,
            low_price,
            volume,
            trades,
            GREATEST(
                high_price - low_price,
                ABS(high_price - LAG(close_price) OVER (ORDER BY start_time)),
                ABS(low_price  - LAG(close_price) OVER (ORDER BY start_time))
            ) AS true_range
        FROM {RAW_VIEW}
    )
    SELECT
        start_time,
        -- SMA
        AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN  4 PRECEDING AND CURRENT ROW) AS sma_5,
        AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN  9 PRECEDING AND CURRENT ROW) AS sma_10,
        AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
        -- Bollinger
        AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            + 2.0 * STDDEV(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            AS bb_upper,
        AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            - 2.0 * STDDEV(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            AS bb_lower,
        (4.0 * STDDEV(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
            / NULLIF(AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
            AS bb_width_pct,
        -- %B: позиция цены внутри BB
        (close_price
            - (AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
               - 2.0 * STDDEV(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)))
            / NULLIF(4.0 * STDDEV(close_price) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
            AS bb_pct_b,
        -- ATR
        AVG(true_range) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS atr_14,
        -- Объём
        volume / NULLIF(AVG(volume) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0)
            AS vol_ratio_20,
        CASE
            WHEN volume > 2.0 * AVG(volume) OVER (ORDER BY start_time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            THEN 1 ELSE 0
        END AS is_high_volume
    FROM tr
),

-- ── Свечные признаки ──────────────────────────────────────────
candle_calc AS (
    SELECT
        start_time,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
        trades,
        ABS(close_price - open_price)
            / NULLIF(high_price - low_price, 0)                         AS body_ratio,
        (high_price - GREATEST(open_price, close_price))
            / NULLIF(high_price - low_price, 0)                         AS upper_wick_ratio,
        (LEAST(open_price, close_price) - low_price)
            / NULLIF(high_price - low_price, 0)                         AS lower_wick_ratio,
        (close_price - low_price)
            / NULLIF(high_price - low_price, 0)                         AS close_position,
        CASE WHEN close_price > open_price THEN 1 ELSE 0 END            AS is_bullish,
        (close_price - open_price) / NULLIF(open_price, 0) * 100.0      AS candle_return_pct,
        (open_price - LAG(close_price) OVER (ORDER BY start_time))
            / NULLIF(LAG(close_price) OVER (ORDER BY start_time), 0) * 100.0
                                                                         AS gap_pct,
        volume / NULLIF(CAST(trades AS DOUBLE), 0)                      AS vol_per_trade
    FROM {RAW_VIEW}
)

-- ── Финальный SELECT ──────────────────────────────────────────
SELECT
    -- Идентификация
    c.start_time,
    CAST(c.start_time AS DATE)      AS trade_date,
    HOUR(c.start_time)              AS hour_of_day,
    DAY_OF_WEEK(c.start_time)       AS day_of_week,

    -- OHLCV
    c.open_price,
    c.high_price,
    c.low_price,
    c.close_price,
    c.volume,
    c.trades,

    -- Свечные признаки
    c.body_ratio,
    c.upper_wick_ratio,
    c.lower_wick_ratio,
    c.close_position,
    c.is_bullish,
    c.candle_return_pct,
    c.gap_pct,
    c.vol_per_trade,

    -- RSI
    r.rsi_14,
    r.rsi_7,
    CASE
        WHEN r.rsi_14 >= 70 THEN 'overbought'
        WHEN r.rsi_14 <= 30 THEN 'oversold'
        ELSE 'neutral'
    END AS rsi_zone,

    -- MACD
    m.macd,
    m.macd_signal,
    ROUND(m.macd - m.macd_signal, 6) AS macd_hist,

    -- Bollinger + SMA + ATR + Объём
    b.sma_5,
    b.sma_10,
    b.sma_20,
    b.bb_upper,
    b.bb_lower,
    b.bb_width_pct,
    b.bb_pct_b,
    b.atr_14,
    ROUND(b.atr_14 / NULLIF(c.close_price, 0) * 100.0, 4) AS atr_pct,
    b.vol_ratio_20,
    b.is_high_volume,

    -- Цвет свечи для Superset
    CASE WHEN c.close_price >= c.open_price
         THEN '#26a69a' ELSE '#ef5350'
    END AS candle_color,

    -- Предсказания модели (NULL если ещё не записаны)
    p.pred_dir_1h,
    p.pred_close_1h,
    p.pred_proba_up,
    p.tp_level,
    p.sl_level,
    COALESCE(p.signal, 'N/A') AS signal,
    p.model_version,
    COALESCE(p.signal, '—')   AS trade_signal

FROM candle_calc c
LEFT JOIN rsi_calc  r ON r.start_time = c.start_time
LEFT JOIN macd_calc m ON m.start_time = c.start_time
LEFT JOIN bb_calc   b ON b.start_time = c.start_time
LEFT JOIN {PRED_TABLE} p ON p.start_time = c.start_time

ORDER BY c.start_time
    """, "[1/2] Создание v_superset_dashboard...")


def verify_view():
    """Проверяет что VIEW создан и возвращает данные."""
    print("[2/2] Проверка VIEW...")
    cur.execute(f"""
        SELECT
            COUNT(*)          AS total_rows,
            MIN(start_time)   AS date_from,
            MAX(start_time)   AS date_to,
            COUNT(rsi_14)     AS rsi_not_null,
            COUNT(macd)       AS macd_not_null,
            COUNT(bb_upper)   AS bb_not_null,
            COUNT(signal)     AS signals_not_null
        FROM {CATALOG}.{FEATURE_SCHEMA}.v_superset_dashboard
    """)
    row = cur.fetchone()
    desc = [d[0] for d in cur.description]
    print("\n    Результат:")
    for col, val in zip(desc, row):
        print(f"      {col:<22} {val}")


def main():
    print("\n" + "="*55)
    print("  Создание v_superset_dashboard")
    print("="*55 + "\n")

    drop_view()
    create_superset_dashboard_view()
    verify_view()

    print(f"\n✓ VIEW готов: {CATALOG}.{FEATURE_SCHEMA}.v_superset_dashboard")
    print("\nПодключи в Superset:")
    print(f"  Dataset → + Dataset → Database: Trino")
    print(f"  Schema: {FEATURE_SCHEMA}")
    print(f"  Table:  v_superset_dashboard")


if __name__ == "__main__":
    main()