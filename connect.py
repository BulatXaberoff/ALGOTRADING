import boto3
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import io
from datetime import date

URL = "https://iss.moex.com/iss/index.xml"
BUCKET = "datalake"
SOURCE = "moex"
DOMAIN = "metadata"
LOAD_DATE = date.today().isoformat()

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="admin",
    aws_secret_access_key="StrongPassword_123!",
    region_name="us-east-1",
)

try:
    s3.create_bucket(Bucket=BUCKET)
except s3.exceptions.BucketAlreadyOwnedByYou:
    pass

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
    # Write to Parquet (in-memory)
    # =========================
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    # =========================
    # S3 path (bronze layer)
    # =========================
    s3_key = (
        f"bronze/{SOURCE}/{DOMAIN}/{dataset_id}/"
        f"load_date={LOAD_DATE}/"
        f"{dataset_id}.parquet"
    )

    # =========================
    # Upload to MinIO
    # =========================
    s3.put_object(
        Bucket=BUCKET,
        Key=s3_key,
        Body=buffer.getvalue(),
    )

    print(f"Uploaded: s3://{BUCKET}/{s3_key}")
