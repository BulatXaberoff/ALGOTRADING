import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.ru1.storage.beget.cloud",
    aws_access_key_id="1T59WAW1TM38Q0BC3R6J",
    aws_secret_access_key="ZmSCe0EXUKbhf56gx8UdOj7DCBWyy3ZbDVXLjw3C",
)

print(s3.list_objects_v2(Bucket="3efb7dce35fe-firm-taelor"))