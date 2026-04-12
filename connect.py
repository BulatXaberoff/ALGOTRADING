import boto3
from botocore.client import Config
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import io
from datetime import date
import os
from dotenv import load_dotenv
from pathlib import Path

# =========================
# Load .env (🔥 ОБНОВЛЕНО)
# =========================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def get_env(name):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"❌ Переменная {name} не найдена")
    return value

S3_ENDPOINT = get_env("S3_ENDPOINT")
S3_BUCKET = get_env("S3_BUCKET")
S3_ACCESS_KEY = get_env("S3_ACCESS_KEY")
S3_SECRET_KEY = get_env("S3_SECRET_KEY")

# =========================
# S3 client (рабочий)
# =========================
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(
        signature_version="s3",
        s3={"addressing_style": "path"}
    ),
)

# =========================
# Проверка S3
# =========================
try:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key="test_connection.txt",
        Body=b"ok",
        ContentLength=2
    )
    print("✅ S3 доступен")
except Exception as e:
    raise RuntimeError(f"❌ Ошибка доступа к S3: {e}")

# =========================
# Constants
# =========================
URL = "https://iss.moex.com/iss/index.xml"
SOURCE = "moex"
DOMAIN = "metadata"
LOAD_DATE = date.today().isoformat()

# =========================
# Load MOEX XML
# =========================
response = requests.get(URL)
response.raise_for_status()

root = ET.fromstring(response.content)

# =========================
# Parse datasets
# =========================
for data in root.findall("./data"):
    dataset_id = data.attrib["id"]

    columns = [
        c.attrib["name"]
        for c in data.findall("./metadata/columns/column")
    ]

    rows = data.findall("./rows/row")

    records = [
        {col: row.attrib.get(col) for col in columns}
        for row in rows
    ]

    if not records:
        continue

    df = pd.DataFrame(records)

    # =========================
    # Convert to parquet
    # =========================
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)

    data_bytes = buffer.getvalue()

    # =========================
    # S3 path
    # =========================
    s3_key = (
        f"bronze/{SOURCE}/{DOMAIN}/{dataset_id}/"
        f"load_date={LOAD_DATE}/"
        f"{dataset_id}.parquet"
    )

    # =========================
    # Upload
    # =========================
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=data_bytes,
            ContentLength=len(data_bytes)
        )
        print(f"✅ Uploaded: s3://{S3_BUCKET}/{s3_key}")

    except Exception as e:
        print(f"❌ Ошибка загрузки {dataset_id}: {e}")

print("\n🎯 Загрузка завершена")