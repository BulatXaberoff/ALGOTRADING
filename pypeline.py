"""
Инициализация структуры каталогов в S3 и инкрементальная загрузка Parquet в Trino.

  python pipeline.py init              — создать «папки» и state в S3 (по данным MOEX ISS)
  python pipeline.py sync              — догрузить в Trino только новые load_date (состояние в S3)

Переменные: как в connect.py / fetch_from_s3.py (.env).
Дополнительно: TRINO_DEFAULT_SCHEMA (по умолчанию web), TRINO_DEFAULT_CATALOG (hive).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import requests
import xml.etree.ElementTree as ET
from botocore.exceptions import ClientError

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Импорт из соседнего модуля (общий S3/Trino)
from fetch_from_s3 import (
    dataframe_to_trino,
    make_s3_client,
    read_parquet_key,
    _require,
)

MOEX_INDEX_URL = os.getenv("MOEX_INDEX_URL", "https://iss.moex.com/iss/index.xml")

STATE_KEY = os.getenv("PIPELINE_STATE_KEY", "_pipeline/state/load_state.json")

# Префиксы «корней» дата-озера (маркеры-пустышки)
# Ключи-маркеры (оканчиваются на / — превращаем в …/.keep для явного «каталога»)
DEFAULT_ROOT_MARKERS = [
    "bronze",
    "bronze/moex",
    "bronze/moex/metadata",
    "silver",
    "gold",
    "_pipeline",
    "_pipeline/state",
]


def discover_dataset_ids_from_moex() -> list[str]:
    r = requests.get(MOEX_INDEX_URL, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ids: list[str] = []
    for data in root.findall("./data"):
        did = data.attrib.get("id")
        if did:
            ids.append(did)
    return sorted(set(ids))


def init_s3_structure() -> None:
    client = make_s3_client()
    bucket = _require("S3_BUCKET")

    print("Загрузка списка наборов с MOEX ISS…")
    try:
        dataset_ids = discover_dataset_ids_from_moex()
    except Exception as e:
        print(f"Не удалось получить index.xml: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Найдено наборов: {len(dataset_ids)}")

    for key in DEFAULT_ROOT_MARKERS:
        k = key if key.endswith(".keep") else key.rstrip("/") + "/.keep"
        client.put_object(Bucket=bucket, Key=k, Body=b"")
        print(f"  маркер: s3://{bucket}/{k}")

    for did in dataset_ids:
        key = f"bronze/moex/metadata/{did}/.init"
        client.put_object(Bucket=bucket, Key=key, Body=b"")
        print(f"  набор: bronze/moex/metadata/{did}/")

    state: dict[str, Any] = {
        "version": 1,
        "initialized_at": date.today().isoformat(),
        "moex_url": MOEX_INDEX_URL,
        "datasets": {},
    }
    client.put_object(
        Bucket=bucket,
        Key=STATE_KEY,
        Body=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"Состояние: s3://{bucket}/{STATE_KEY}")
    print("Готово: структура S3 и начальный state созданы.")


def load_state(client, bucket: str) -> dict[str, Any]:
    try:
        r = client.get_object(Bucket=bucket, Key=STATE_KEY)
        return json.loads(r["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return {"version": 1, "datasets": {}}
        raise


def save_state(client, bucket: str, state: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=STATE_KEY,
        Body=json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


PARQUET_PATH_RE = re.compile(
    r"^bronze/moex/metadata/(?P<ds>[^/]+)/load_date=(?P<ld>[\d-]+)/.+\.parquet$"
)


def list_all_parquet_keys(client, bucket: str, prefix: str = "bronze/moex/metadata/") -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(".parquet"):
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return keys


def parse_key(key: str) -> tuple[str, str] | None:
    m = PARQUET_PATH_RE.match(key)
    if not m:
        return None
    return m.group("ds"), m.group("ld")


def sanitize_table_name(dataset_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", dataset_id)
    if s and s[0].isdigit():
        s = "t_" + s
    return s.lower() or "dataset"


def incremental_sync_to_trino(
    *,
    catalog: str,
    schema: str,
    trino_batch_size: int,
    dry_run: bool,
) -> None:
    client = make_s3_client()
    bucket = _require("S3_BUCKET")
    state = load_state(client, bucket)
    datasets_state: dict[str, str] = state.setdefault("datasets", {})

    keys = list_all_parquet_keys(client, bucket)
    parsed: list[tuple[str, str, str]] = []
    for k in keys:
        p = parse_key(k)
        if not p:
            continue
        ds, ld = p
        parsed.append((ds, ld, k))

    parsed.sort(key=lambda x: (x[0], x[1], x[2]))

    pending: list[tuple[str, str, str]] = []
    for ds, ld, k in parsed:
        last = datasets_state.get(ds, "0000-01-01")
        if ld > last:
            pending.append((ds, ld, k))

    if not pending:
        print("Новых parquet относительно state нет — синхронизация не требуется.")
        return

    print(f"К загрузке файлов: {len(pending)}")
    if dry_run:
        for ds, ld, k in pending:
            print(f"  [dry-run] {ds} load_date={ld} → {k}")
        return

    for ds, ld, k in pending:
        table = sanitize_table_name(ds)
        full_table = f"{catalog}.{schema}.{table}"
        print(f"Загрузка {k} → {full_table} …")
        df = read_parquet_key(client, bucket, k)
        # первый файл набора в истории sync — создать таблицу; следующие load_date — append
        if_exists_mode = "append" if ds in datasets_state else "replace"
        dataframe_to_trino(
            df,
            full_table,
            if_exists=if_exists_mode,
            batch_size=trino_batch_size,
        )
        # после успешной вставки обновляем watermark по дате (макс. для набора)
        prev = datasets_state.get(ds, "0000-01-01")
        datasets_state[ds] = max(prev, ld)
        save_state(client, bucket, state)
        print(f"  state[{ds}] = {datasets_state[ds]}")

    print("Инкрементальная синхронизация завершена.")


def main() -> None:
    p = argparse.ArgumentParser(description="S3 layout init + инкрементальная загрузка в Trino")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Создать структуру каталогов в S3 и начальный state")

    ps = sub.add_parser("sync", help="Загрузить в Trino новые parquet относительно state")
    ps.add_argument("--catalog", default=os.getenv("TRINO_DEFAULT_CATALOG", "hive"))
    ps.add_argument("--schema", default=os.getenv("TRINO_DEFAULT_SCHEMA", "web"))
    ps.add_argument("--trino-batch-size", type=int, default=500)
    ps.add_argument("--dry-run", action="store_true", help="Только показать, что было бы загружено")

    args = p.parse_args()

    if args.cmd == "init":
        init_s3_structure()
        return

    if args.cmd == "sync":
        incremental_sync_to_trino(
            catalog=args.catalog,
            schema=args.schema,
            trino_batch_size=args.trino_batch_size,
            dry_run=args.dry_run,
        )
        return


if __name__ == "__main__":
    main()
