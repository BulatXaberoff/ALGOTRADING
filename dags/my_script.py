# candles_loader.py
# Загрузка свечей MOEX (1H) → S3 (MinIO) в формате Parquet

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import boto3
from moexalgo import Ticker

# =========================
# Constants & Config
# =========================
SOURCE = "moex"
DOMAIN = "candles"

SECIDS = ["GAZP", "SBER", "LKOH"]

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now()

PERIOD = 60
TIMEFRAME = timedelta(hours=1)

RUN_TIMESTAMP = datetime.now()
LOAD_TIMESTAMP_STR = RUN_TIMESTAMP.strftime("%Y%m%d_%H%M%S")
LOAD_TIMESTAMP_VAL = RUN_TIMESTAMP.strftime("%Y-%m-%d %H:%M:%S")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "moex-data" / SOURCE / DOMAIN

# S3 / MinIO
S3_BUCKET = "data"
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minio_access_key",
    aws_secret_access_key="minio_secret_key",
    region_name="us-east-1"
)

# =========================
# Load Candles
# =========================
def fetch_candles(secid: str) -> pd.DataFrame:
    print(f"📥 Загружаем {secid}")

    date = START_DATE
    all_data = []

    while True:
        data = list(
            Ticker(secid).candles(
                date=date,
                till_date=END_DATE,
                period=PERIOD,
                limit=50000,
            )
        )

        if not data:
            break

        df = pd.DataFrame(
            data,
            columns=[
                "begin",
                "end",
                "open",
                "high",
                "low",
                "close",
                "value",
                "volume",
            ],
        )

        all_data.append(df)

        last_time = pd.to_datetime(df.iloc[-1]["begin"])
        date = last_time + TIMEFRAME

        if len(df) < 50000:
            break

    if not all_data:
        return pd.DataFrame()

    df_final = pd.concat(all_data, ignore_index=True)
    return df_final


# =========================
# Save Dataset
# =========================
def save_dataset(
    secid: str,
    df: pd.DataFrame,
    base_path: Path,
    timestamp_val: str,
    timestamp_str: str,
    s3_client,
    bucket: str,
):
    if df.empty:
        print(f"⚠️ Нет данных для {secid}")
        return

    # 🔥 служебные поля
    df["load_timestamp"] = timestamp_val
    df["source"] = SOURCE
    df["instrument"] = secid

    # строки (как в bronze best practice)
    for col in df.columns:
        df[col] = df[col].astype("string")

    # =========================
    # Local save
    # =========================
    dataset_path = base_path / secid
    dataset_path.mkdir(parents=True, exist_ok=True)

    filename = f"{secid}_1h_{timestamp_str}.parquet"
    parquet_path = dataset_path / filename

    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    # =========================
    # Upload to S3
    # =========================
    s3_key = f"moex-data/{SOURCE}/{DOMAIN}/{secid}/{filename}"

    s3_client.upload_file(str(parquet_path), bucket, s3_key)

    print(f"✅ [{secid}] сохранен")
    print(f"☁️ S3: s3://{bucket}/{s3_key}")
    print("-" * 60)


# =========================
# Main Pipeline
# =========================
def main():
    print(f"🚀 Старт загрузки свечей | {LOAD_TIMESTAMP_VAL}")

    processed = 0

    for secid in SECIDS:
        df = fetch_candles(secid)

        save_dataset(
            secid=secid,
            df=df,
            base_path=DATA_PATH,
            timestamp_val=LOAD_TIMESTAMP_VAL,
            timestamp_str=LOAD_TIMESTAMP_STR,
            s3_client=s3,
            bucket=S3_BUCKET,
        )

        processed += 1

    print(f"\n🎯 Готово. Обработано инструментов: {processed}")


if __name__ == "__main__":
    main()