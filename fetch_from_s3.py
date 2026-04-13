"""
S3: список объектов, чтение Parquet.
Опционально: загрузка DataFrame в Trino без ручного SQL (CREATE TABLE + INSERT батчами).

.env: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET; опционально S3_REGION.
Trino: TRINO_HOST, TRINO_PORT, TRINO_USER, TRINO_HTTP_SCHEME (http|https); пакет trino.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"Ошибка: в .env не задана переменная {name}", file=sys.stderr)
        sys.exit(1)
    return v


def make_s3_client():
    endpoint = _require("S3_ENDPOINT")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=_require("S3_ACCESS_KEY"),
        aws_secret_access_key=_require("S3_SECRET_KEY"),
        config=Config(
            signature_version="s3",
            s3={"addressing_style": "path"},
        ),
    )


def list_objects(client, bucket: str, prefix: str, max_keys: int = 500) -> list[dict]:
    """Возвращает список dict с ключами Key, Size, LastModified (как в list_objects_v2)."""
    out: list[dict] = []
    token = None
    while len(out) < max_keys:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": min(1000, max_keys - len(out))}
        if token:
            kwargs["ContinuationToken"] = token
        try:
            resp = client.list_objects_v2(**kwargs)
        except ClientError as e:
            print(f"list_objects_v2 отказ: {e}", file=sys.stderr)
            raise
        for obj in resp.get("Contents", []):
            out.append(
                {
                    "Key": obj["Key"],
                    "Size": obj.get("Size"),
                    "LastModified": str(obj.get("LastModified")),
                }
            )
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def get_object_bytes(client, bucket: str, key: str) -> bytes:
    try:
        r = client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        print(f"get_object отказ для {key!r}: {e}", file=sys.stderr)
        raise
    return r["Body"].read()


def read_parquet_key(client, bucket: str, key: str) -> pd.DataFrame:
    raw = get_object_bytes(client, bucket, key)
    return pd.read_parquet(io.BytesIO(raw))


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _trino_type_for_series(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "bigint"
    if pd.api.types.is_float_dtype(s):
        return "double"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "timestamp(6)"
    return "varchar"


def _sql_literal(val, col_sql_type: str) -> str:
    try:
        if val is None or pd.isna(val):
            return "NULL"
    except (ValueError, TypeError):
        if val is None:
            return "NULL"
    if isinstance(val, float) and math.isnan(val):
        return "NULL"
    if col_sql_type == "boolean":
        return "TRUE" if bool(val) else "FALSE"
    if col_sql_type == "bigint":
        return str(int(val))
    if col_sql_type == "double":
        return repr(float(val))
    if col_sql_type == "timestamp(6)":
        ts = pd.Timestamp(val)
        return "TIMESTAMP '" + ts.strftime("%Y-%m-%d %H:%M:%S.%f") + "'"
    s = str(val)
    return "'" + s.replace("'", "''").replace("\\", "\\\\") + "'"


def parse_trino_table(full_name: str) -> tuple[str, str, str]:
    """
    Полное имя: catalog.schema.table (ровно три компонента через точку).
    """
    parts = full_name.strip().split(".")
    if len(parts) != 3:
        print(
            "Ожидается имя вида catalog.schema.table, например hive.web.securitygroups",
            file=sys.stderr,
        )
        sys.exit(1)
    if not all(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", p) for p in parts):
        print("Недопустимые символы в имени каталога/схемы/таблицы.", file=sys.stderr)
        sys.exit(1)
    return parts[0], parts[1], parts[2]


def _trino_connect(catalog: str, schema: str):
    try:
        import trino.dbapi as trino_dbapi
    except ImportError:
        print("Установите пакет: pip install trino", file=sys.stderr)
        sys.exit(1)

    host = os.getenv("TRINO_HOST", "localhost")
    port = int(os.getenv("TRINO_PORT", "8080"))
    user = os.getenv("TRINO_USER", "admin")
    scheme = os.getenv("TRINO_HTTP_SCHEME", "http").lower()
    if scheme not in ("http", "https"):
        scheme = "http"

    return trino_dbapi.connect(
        host=host,
        port=port,
        user=user,
        catalog=catalog,
        schema=schema,
        http_scheme=scheme,
    )


def dataframe_to_trino(
    df: pd.DataFrame,
    full_table: str,
    *,
    if_exists: str = "replace",
    batch_size: int = 500,
) -> None:
    """
    Создаёт схему при необходимости (CREATE SCHEMA IF NOT EXISTS), затем таблицу по типам pandas
    и вставляет строки батчами (INSERT ... VALUES).
    if_exists: fail | replace | append
    """
    catalog, schema, table = parse_trino_table(full_table)
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].replace({float("nan"): None})

    col_types = {c: _trino_type_for_series(df[c]) for c in df.columns}
    col_list_sql = ", ".join(f"{_quote_ident(c)} {col_types[c]}" for c in df.columns)
    fq = f"{_quote_ident(catalog)}.{_quote_ident(schema)}.{_quote_ident(table)}"
    cols_sql = ", ".join(_quote_ident(c) for c in df.columns)

    conn = _trino_connect(catalog, schema)
    cur = None
    try:
        cur = conn.cursor()
        schema_fq = f"{_quote_ident(catalog)}.{_quote_ident(schema)}"
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_fq}")

        if if_exists == "replace":
            cur.execute(f"DROP TABLE IF EXISTS {fq}")
            cur.execute(f"CREATE TABLE {fq} ({col_list_sql})")
        elif if_exists == "fail":
            cur.execute(f"CREATE TABLE {fq} ({col_list_sql})")
        else:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {fq} ({col_list_sql})")

        if len(df) == 0:
            print("DataFrame пустой — таблица создана, INSERT не выполнялся.")
            return

        rows = df.to_dict(orient="records")
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            values_sql = []
            for rec in chunk:
                literals = [_sql_literal(rec.get(c), col_types[c]) for c in df.columns]
                values_sql.append("(" + ", ".join(literals) + ")")
            sql = f"INSERT INTO {fq} ({cols_sql}) VALUES " + ", ".join(values_sql)
            cur.execute(sql)

        print(f"В Trino загружено строк: {len(df)} → {full_table}")
    finally:
        if cur is not None:
            cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="S3: список/чтение Parquet; опционально загрузка в Trino")
    parser.add_argument(
        "--prefix",
        default="bronze/moex/metadata/securitygroups/",
        help="Префикс внутри бакета",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Имя бакета (по умолчанию S3_BUCKET из .env)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Конкретный ключ Parquet в бакете",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Только список объектов по префиксу",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=200,
        help="Максимум объектов в списке",
    )
    parser.add_argument(
        "--to-trino-table",
        default=None,
        metavar="catalog.schema.table",
        help="После чтения Parquet — создать таблицу и вставить данные (pip install trino)",
    )
    parser.add_argument(
        "--trino-if-exists",
        choices=("fail", "replace", "append"),
        default="replace",
        help="Поведение, если таблица в Trino уже есть (по умолчанию replace)",
    )
    parser.add_argument(
        "--trino-batch-size",
        type=int,
        default=500,
        help="Строк в одном INSERT",
    )
    args = parser.parse_args()

    bucket = args.bucket or _require("S3_BUCKET")
    client = make_s3_client()

    if args.list_only and args.to_trino_table:
        print("Нельзя совмещать --list-only и --to-trino-table.", file=sys.stderr)
        sys.exit(1)

    if args.key:
        df = read_parquet_key(client, bucket, args.key)
    else:
        objects = list_objects(client, bucket, args.prefix, max_keys=args.max_keys)
        if not objects:
            print(f"Объектов с префиксом {args.prefix!r} не найдено.")
            return
        print(f"Бакет: {bucket}, префикс: {args.prefix!r}, объектов: {len(objects)}")
        for o in objects:
            print(f"  {o['Key']}\t size={o['Size']}")
        if args.list_only:
            return
        parquet_keys = [o["Key"] for o in objects if o["Key"].lower().endswith(".parquet")]
        if not parquet_keys:
            print("\nParquet-файлов по префиксу нет — укажите --key.")
            return
        key = parquet_keys[0]
        if len(parquet_keys) > 1:
            print(f"\nНесколько parquet; читаем первый: {key!r}")
        df = read_parquet_key(client, bucket, key)

    print("\n--- Первые строки ---")
    print(df.head(20).to_string())
    print(f"\nСтрок: {len(df)}, колонок: {len(df.columns)}")

    if args.to_trino_table:
        dataframe_to_trino(
            df,
            args.to_trino_table,
            if_exists=args.trino_if_exists,
            batch_size=args.trino_batch_size,
        )


if __name__ == "__main__":
    main()
