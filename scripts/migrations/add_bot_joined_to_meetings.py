#!/usr/bin/env python3
"""
Migration: Add/Backfill `bot_joined` attribute on meetings items in DynamoDB.

This script scans the meetings table and sets `bot_joined` to False where the
attribute is missing. No table schema change is required in DynamoDB for
non-key attributes.

Usage:
  python -m clerk_backend.scripts.migrations.add_bot_joined_to_meetings --yes

Optional:
  --region <AWS_REGION>
  --endpoint <DYNAMODB_ENDPOINT_URL>   # e.g., http://localhost:8000 for DynamoDB Local
  --table <TABLE_NAME>
  --batch-size 50
"""
import argparse
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from clerk_backend.shared.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dynamodb_resource(region: Optional[str], endpoint: Optional[str]):
    settings = get_settings()
    region_name = region or settings.aws_region

    if endpoint:
        logger.info(f"Using DynamoDB endpoint override: {endpoint}")
        return boto3.resource("dynamodb", region_name=region_name, endpoint_url=endpoint)

    if settings.aws_access_key_id:
        return boto3.resource(
            "dynamodb",
            region_name=region_name,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
    else:
        # Default to local DynamoDB if no creds configured
        endpoint_url = os.getenv("DYNAMODB_LOCAL_ENDPOINT", "http://localhost:8000")
        logger.info(f"Using DynamoDB Local at {endpoint_url}")
        return boto3.resource("dynamodb", region_name=region_name, endpoint_url=endpoint_url)


def resolve_meetings_table_name(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    settings = get_settings()
    return f"{settings.dynamodb_table_prefix}{settings.meetings_table}"


def backfill_bot_joined(table, batch_size: int = 50) -> int:
    """
    Scan the table and set bot_joined = false if attribute is missing.
    Returns the number of items updated.
    """
    import time

    updated = 0
    start_key = None

    while True:
        scan_kwargs = {
            "Limit": max(25, batch_size),
            "ProjectionExpression": "#id, bot_joined",
            "ExpressionAttributeNames": {"#id": "id"},
        }
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key

        resp = table.scan(**scan_kwargs)
        items = resp.get("Items", [])

        for item in items:
            key = {"id": item["id"]}
            # Set only if missing
            try:
                table.update_item(
                    Key=key,
                    UpdateExpression="SET bot_joined = if_not_exists(bot_joined, :b)",
                    ExpressionAttributeValues={":b": False},
                )
                updated += 1
            except ClientError as e:
                logger.error(f"Update failed for {key['id']}: {e}")

        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break

        # Be nice to provisioned throughput
        time.sleep(0.1)

    return updated


def main():
    parser = argparse.ArgumentParser(description="Backfill bot_joined on meetings items")
    parser.add_argument("--yes", action="store_true", help="Run without interactive prompt")
    parser.add_argument("--region", default=None, help="AWS region override")
    parser.add_argument("--endpoint", default=None, help="DynamoDB endpoint override (e.g., http://localhost:8000)")
    parser.add_argument("--table", default=None, help="Meetings table name override")
    parser.add_argument("--batch-size", type=int, default=50, help="Scan batch size")
    args = parser.parse_args()

    if not args.yes:
        print("This will scan and update all meetings to ensure bot_joined exists (default False).")
        confirm = input("Proceed? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    dynamodb = build_dynamodb_resource(args.region, args.endpoint)
    table_name = resolve_meetings_table_name(args.table)
    table = dynamodb.Table(table_name)
    logger.info(f"Backfilling bot_joined on table: {table_name}")

    try:
        count = backfill_bot_joined(table, batch_size=args.batch_size)
        logger.info(f"Backfill complete. Items touched: {count}")
        print({"success": True, "updated": count, "table": table_name})
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()


