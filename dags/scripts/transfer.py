import trino
import os

conn = trino.dbapi.connect(
    host=os.getenv('TRINO_HOST', 'trino'),
    port=int(os.getenv('TRINO_PORT', 8080)),
    user='admin',
    catalog="minio",
    schema="moex_data",
)

cur = conn.cursor()

CATALOG = "minio"
SCHEMA  = "moex_data"
TABLE   = "GAZP"

# Маппинг колонок -> типы
COLUMN_TYPE_MAP = {
    "start_time":  "TIMESTAMP",
    "end_time":    "TIMESTAMP",
    "open_price":  "REAL",
    "high_price":  "REAL",
    "low_price":   "REAL",
    "close_price": "REAL",
    "volume":      "REAL",
    "trades":      "REAL",
}

EXTERNAL_LOCATION = f"s3a://data/moex-data/moex/candles/{TABLE}"


def recreate_external_table():
    """Пересоздаёт внешнюю таблицу поверх S3"""
    full_table = f"{CATALOG}.{SCHEMA}.{TABLE}"
    print(f"[1/2] Пересоздаём внешнюю таблицу {full_table}...")

    cur.execute(f"DROP TABLE IF EXISTS {full_table}")

    cur.execute(f"""
        CREATE TABLE {full_table} (
            start_time  VARCHAR,
            end_time    VARCHAR,
            open_price  VARCHAR,
            high_price  VARCHAR,
            low_price   VARCHAR,
            close_price VARCHAR,
            volume      VARCHAR,
            trades      VARCHAR
        )
        WITH (
            format = 'PARQUET',
            external_location = '{EXTERNAL_LOCATION}'
        )
    """)
    print(f"    ✓ Таблица {full_table} создана")


def recreate_view():
    """Пересоздаёт VIEW с кастом колонок в правильные типы"""
    source    = f"{CATALOG}.{SCHEMA}.{TABLE}"
    view_name = f"{CATALOG}.{SCHEMA}.{TABLE}_view"
    print(f"[2/2] Пересоздаём VIEW {view_name}...")

    # Получаем актуальный список колонок из таблицы
    cur.execute(f"DESCRIBE {source}")
    columns = [row[0] for row in cur.fetchall()]

    # Строим SELECT с кастами
    select_parts = []
    for col in columns:
        target_type = COLUMN_TYPE_MAP.get(col)
        if target_type:
            select_parts.append(
                f"TRY_CAST({col} AS {target_type}) AS {col}"
            )
        else:
            select_parts.append(col)

    select_clause = ",\n            ".join(select_parts)

    cur.execute(f"DROP VIEW IF EXISTS {view_name}")
    cur.execute(f"""
        CREATE VIEW {view_name} AS
        SELECT
            {select_clause}
        FROM {source}
        GROUP BY start_time, end_time, open_price, high_price, low_price, close_price, volume, trades
    """)
    print(f"    ✓ VIEW {view_name} создана ({len(columns)} колонок)")


def main():
    recreate_external_table()
    recreate_view()
    print("\nГотово! Используйте:")
    print(f"  SELECT * FROM {CATALOG}.{SCHEMA}.{TABLE}_view LIMIT 10")


main()