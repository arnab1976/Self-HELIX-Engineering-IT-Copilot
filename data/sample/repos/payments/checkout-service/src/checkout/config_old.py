# payments/checkout-service/config.py
# owner: priya.raman@acme-corp.com  on-call: +91 98450 12345
# NOTE: deliberately dirty sample used to demonstrate the LEXORA hygiene pipeline.
# The key below is AWS's public documentation example, not a real credential.
import boto3

AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
DB_PASSWORD = "Summer2026!"
REGION = "ap-south-2"


def get_client():
    return boto3.client("s3", region_name=REGION)
