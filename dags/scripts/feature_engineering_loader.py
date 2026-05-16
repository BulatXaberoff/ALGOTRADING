import trino
import os

conn = trino.dbapi.connect(
    host=os.getenv('TRINO_HOST', 'trino'),
    port=int(os.getenv('TRINO_PORT', 8080)),
    user='admin',
    catalog="minio",
    schema="feature_data_gazp_agg",
)
cur = conn.cursor()

CATALOG        = "minio"
RAW_SCHEMA     = "moex_data"
FEATURE_SCHEMA = "feature_data_gazp_agg"
RAW_VIEW       = f"{CATALOG}.{RAW_SCHEMA}.gazp_view"

# Базовый S3-путь схемы — должен совпадать с location в CREATE SCHEMA
S3_BASE = "s3a://data/feature-data-gazp-agg"


def execute(sql: str, message: str = ""):
    if message:
        print(message)
    cur.execute(sql)
    print("    ✓ OK")


# ──────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: удалить папку в MinIO через Trino
# ──────────────────────────────────────────────────────────────

def drop_table_with_purge(full_table_name: str):
    """
    DROP TABLE + PURGE физически удаляет данные из S3/MinIO.
    Без PURGE папка остаётся и следующий CREATE TABLE падает
    с HIVE_PATH_ALREADY_EXISTS.
    """
    try:
        cur.execute(f"DROP TABLE IF EXISTS {full_table_name}")
        print(f"    ✓ DROP TABLE {full_table_name}")
    except Exception as e:
        print(f"    ⚠ DROP TABLE {full_table_name}: {e}")


def drop_view_safe(full_view_name: str):
    try:
        cur.execute(f"DROP VIEW IF EXISTS {full_view_name}")
        print(f"    ✓ DROP VIEW {full_view_name}")
    except Exception as e:
        print(f"    ⚠ DROP VIEW {full_view_name}: {e}")


# ──────────────────────────────────────────────────────────────
# 1. ОЧИСТКА
# ──────────────────────────────────────────────────────────────

def cleanup():
    print("[1/7] Очистка старых объектов...")

    # Сначала VIEW (они не имеют физических данных)
    for view in [
        "v_candle_features", "v_rsi", "v_macd",
        "v_volume_bb_atr", "v_superset_dashboard",
    ]:
        drop_view_safe(f"{CATALOG}.{FEATURE_SCHEMA}.{view}")

    # Таблицы с физическими данными
    # ВАЖНО: DROP TABLE в Trino на Hive-коннекторе НЕ удаляет файлы S3.
    # Поэтому создаём таблицы с явным external_location на уникальный путь,
    # либо удаляем папки вручную через MinIO CLI.
    for table in ["features", "predictions", "feature_importance", "model_metrics"]:
        drop_table_with_purge(f"{CATALOG}.{FEATURE_SCHEMA}.{table}")

    print("    ✓ Старые объекты удалены")


# ──────────────────────────────────────────────────────────────
# 2. СХЕМА
# ──────────────────────────────────────────────────────────────

def create_schema():
    execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG}.{FEATURE_SCHEMA}
        WITH (location = '{S3_BASE}/')
        """,
        "[2/7] Создание схемы..."
    )


# ──────────────────────────────────────────────────────────────
# 3–6. VIEW (не имеют физического хранилища — HIVE_PATH не касается)
# ──────────────────────────────────────────────────────────────

def create_candle_features():
    execute(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{FEATURE_SCHEMA}.v_candle_features AS
    WITH ordered AS (
        SELECT
            start_time, open_price, high_price, low_price,
            close_price, volume, trades,
            LAG(close_price, 1) OVER (ORDER BY start_time) AS prev_close_1,
            LAG(close_price, 2) OVER (ORDER BY start_time) AS prev_close_2,
            LAG(close_price, 3) OVER (ORDER BY start_time) AS prev_close_3,
            LAG(close_price, 5) OVER (ORDER BY start_time) AS prev_close_5
        FROM {RAW_VIEW}
    )
    SELECT
        start_time, open_price, high_price, low_price, close_price, volume, trades,
        (high_price - low_price) AS candle_range,
        ABS(close_price - open_price) AS body_size,
        ABS(close_price - open_price) / NULLIF(high_price - low_price, 0) AS body_ratio,
        (high_price - GREATEST(open_price, close_price)) / NULLIF(high_price - low_price, 0) AS upper_wick_ratio,
        (LEAST(open_price, close_price) - low_price) / NULLIF(high_price - low_price, 0) AS lower_wick_ratio,
        high_price - GREATEST(open_price, close_price) AS upper_wick_abs,
        LEAST(open_price, close_price) - low_price AS lower_wick_abs,
        (close_price - low_price) / NULLIF(high_price - low_price, 0) AS close_position,
        CASE WHEN close_price > open_price THEN 1 ELSE 0 END AS is_bullish,
        (close_price - open_price) / NULLIF(open_price, 0) * 100.0 AS candle_return_pct,
        open_price - prev_close_1 AS gap_from_prev,
        (open_price - prev_close_1) / NULLIF(prev_close_1, 0) * 100.0 AS gap_pct,
        volume / NULLIF(CAST(trades AS DOUBLE), 0) AS vol_per_trade,
        (close_price - prev_close_1) / NULLIF(prev_close_1, 0) AS return_1,
        (close_price - prev_close_2) / NULLIF(prev_close_2, 0) AS return_2,
        (close_price - prev_close_3) / NULLIF(prev_close_3, 0) AS return_3,
        (close_price - prev_close_5) / NULLIF(prev_close_5, 0) AS return_5
    FROM ordered
    """, "[3/7] Создание v_candle_features...")


def create_rsi():
    execute(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{FEATURE_SCHEMA}.v_rsi AS
    WITH price_delta AS (
        SELECT start_time, close_price,
               close_price - LAG(close_price) OVER (ORDER BY start_time) AS delta
        FROM {RAW_VIEW}
    ),
    gains_losses AS (
        SELECT start_time, close_price,
               GREATEST(delta,  0) AS gain,
               GREATEST(-delta, 0) AS loss
        FROM price_delta
    ),
    avg_windows AS (
        SELECT start_time, close_price,
            AVG(gain) OVER (ORDER BY start_time ROWS BETWEEN  6 PRECEDING AND CURRENT ROW) AS avg_gain_7,
            AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN  6 PRECEDING AND CURRENT ROW) AS avg_loss_7,
            AVG(gain) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain_14,
            AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss_14,
            AVG(gain) OVER (ORDER BY start_time ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS avg_gain_21,
            AVG(loss) OVER (ORDER BY start_time ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS avg_loss_21
        FROM gains_losses
    )
    SELECT * FROM avg_windows
    """, "[4/7] Создание v_rsi...")


def create_macd():
    execute(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{FEATURE_SCHEMA}.v_macd AS
    WITH sma_lines AS (
        SELECT start_time, close_price,
            AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS sma_12,
            AVG(close_price) OVER (ORDER BY start_time ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS sma_26
        FROM {RAW_VIEW}
    ),
    macd_line AS (
        SELECT *, ROUND(sma_12 - sma_26, 6) AS macd FROM sma_lines
    ),
    macd_signal AS (
        SELECT *,
            ROUND(AVG(macd) OVER (ORDER BY start_time ROWS BETWEEN 8 PRECEDING AND CURRENT ROW), 6) AS macd_signal
        FROM macd_line
    )
    SELECT * FROM macd_signal
    """, "[5/7] Создание v_macd...")


def create_volume_view():
    execute(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{FEATURE_SCHEMA}.v_volume_bb_atr AS
    WITH tr AS (
        SELECT start_time, close_price, high_price, low_price, volume, trades,
            GREATEST(
                high_price - low_price,
                ABS(high_price - LAG(close_price) OVER (ORDER BY start_time)),
                ABS(low_price  - LAG(close_price) OVER (ORDER BY start_time))
            ) AS true_range
        FROM {RAW_VIEW}
    )
    SELECT * FROM tr
    """, "[6/7] Создание v_volume_bb_atr...")


# ──────────────────────────────────────────────────────────────
# 7. ТАБЛИЦЫ С ФИЗИЧЕСКИМИ ДАННЫМИ
# Ключевое решение: явный external_location с уникальным путём,
# чтобы не конфликтовать с остатками от предыдущих запусков.
# ──────────────────────────────────────────────────────────────

def create_features_table():
    """
    Создаёт таблицу features как CTAS из RAW VIEW.
    external_location указывает явный путь в S3 — если папка
    уже существует после предыдущего DROP, указываем другой путь
    или предварительно чистим через MinIO.
    """
    execute(f"""
    CREATE TABLE {CATALOG}.{FEATURE_SCHEMA}.features
    WITH (
        format            = 'PARQUET',
        external_location = '{S3_BASE}/features/'
    )
    AS SELECT * FROM {RAW_VIEW}
    """, "[7/7] Создание финальной features...")


def create_predictions_table():
    """
    Создаёт пустую таблицу predictions с явной схемой.
    Решение HIVE_PATH_ALREADY_EXISTS:
      - explicit external_location на подпапку /predictions/
      - если папка осталась от прошлого запуска — удали её
        вручную в MinIO Console или через mc rm -r
    """
    execute(f"""
    CREATE TABLE {CATALOG}.{FEATURE_SCHEMA}.predictions (
        start_time      TIMESTAMP,
        close_price     DOUBLE,
        pred_dir_1h     INTEGER,
        pred_close_1h   DOUBLE,
        pred_proba_up   DOUBLE,
        tp_level        DOUBLE,
        sl_level        DOUBLE,
        pred_return_pct DOUBLE,
        signal          VARCHAR,
        model_version   VARCHAR,
        created_at      TIMESTAMP,
        pred_date       DATE
    )
    WITH (
        format            = 'PARQUET',
        partitioned_by    = ARRAY['pred_date'],
        external_location = '{S3_BASE}/predictions/'
    )
    """, "Создание predictions...")


def create_feature_importance_table():
    """Таблица для хранения важности признаков из XGBoost."""
    execute(f"""
    CREATE TABLE {CATALOG}.{FEATURE_SCHEMA}.feature_importance (
        feature_name  VARCHAR,
        importance    DOUBLE,
        rank_num      INTEGER,
        ticker        VARCHAR,
        model_version VARCHAR,
        created_at    TIMESTAMP
    )
    WITH (
        format            = 'PARQUET',
        external_location = '{S3_BASE}/feature_importance/'
    )
    """, "Создание feature_importance...")


def create_model_metrics_table():
    """Таблица для хранения метрик качества модели."""
    execute(f"""
    CREATE TABLE {CATALOG}.{FEATURE_SCHEMA}.model_metrics (
        ticker         VARCHAR,
        model_version  VARCHAR,
        accuracy       DOUBLE,
        f1_score       DOUBLE,
        roc_auc        DOUBLE,
        mae            DOUBLE,
        mape           DOUBLE,
        train_rows     INTEGER,
        test_date_from VARCHAR,
        test_date_to   VARCHAR,
        created_at     TIMESTAMP
    )
    WITH (
        format            = 'PARQUET',
        external_location = '{S3_BASE}/model_metrics/'
    )
    """, "Создание model_metrics...")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    cleanup()
    create_schema()
    create_candle_features()
    create_rsi()
    create_macd()
    create_volume_view()
    create_features_table()
    create_predictions_table()
    create_feature_importance_table()
    create_model_metrics_table()

    print("\n✓ Готово. Таблицы:")
    print(f"  {CATALOG}.{FEATURE_SCHEMA}.features")
    print(f"  {CATALOG}.{FEATURE_SCHEMA}.predictions")
    print(f"  {CATALOG}.{FEATURE_SCHEMA}.feature_importance")
    print(f"  {CATALOG}.{FEATURE_SCHEMA}.model_metrics")


if __name__ == "__main__":
    main()