#!/usr/bin/env python3
"""
Script to add missing GSI to existing DynamoDB tables.
Specifically adds user-id-start-time-index to meetings table.
"""
import os
import sys
import logging

# Add the parent directory to the path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

from shared.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


def add_missing_index():
    """Add missing user-id-start-time-index to meetings table."""
    try:
        # Determine if we should use local DynamoDB (same logic as DAO)
        use_local = settings.use_local_dynamodb
        
        # Auto-detect: if USE_LOCAL_DYNAMODB is not explicitly set and AWS credentials are missing,
        # try local DynamoDB first
        if not use_local and not settings.aws_access_key_id:
            logger.info("No AWS credentials found. Using local DynamoDB...")
            use_local = True
        
        # Initialize DynamoDB client
        if use_local:
            logger.info(f"Connecting to local DynamoDB at {settings.dynamodb_local_endpoint}")
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url=settings.dynamodb_local_endpoint,
                aws_access_key_id="dummy",  # Required for local DynamoDB
                aws_secret_access_key="dummy",  # Required for local DynamoDB
            )
        elif settings.aws_access_key_id:
            logger.info(f"Connecting to AWS DynamoDB (Region: {settings.aws_region})")
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            # Try using IAM role (for ECS/Lambda)
            logger.info(f"Connecting to AWS DynamoDB using IAM role (Region: {settings.aws_region})")
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
            )

        table_name = f"{settings.dynamodb_table_prefix}{settings.meetings_table}"
        table = dynamodb.Table(table_name)

        # Check if table exists
        try:
            table.load()
            logger.info(f"Table {table_name} exists")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.error(f"Table {table_name} does not exist. Please create it first.")
                return
            raise

        # Check if index already exists
        existing_indexes = [idx['IndexName'] for idx in table.global_secondary_indexes or []]
        if "user-id-start-time-index" in existing_indexes:
            logger.info(f"Index user-id-start-time-index already exists on {table_name}")
            return

        # Add the missing index
        logger.info(f"Adding user-id-start-time-index to {table_name}...")
        
        # Use client for update_table operation
        if settings.aws_access_key_id:
            client = boto3.client(
                "dynamodb",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            client = boto3.client(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url="http://localhost:8001",
            )

        # Update table to add the GSI
        response = client.update_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "start_time", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexUpdates=[
                {
                    "Create": {
                        "IndexName": "user-id-start-time-index",
                        "KeySchema": [
                            {"AttributeName": "user_id", "KeyType": "HASH"},
                            {"AttributeName": "start_time", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                        "ProvisionedThroughput": {
                            "ReadCapacityUnits": 5,
                            "WriteCapacityUnits": 5,
                        },
                    }
                }
            ],
        )

        logger.info(f"Waiting for index to be created...")
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
        
        logger.info(f"Index user-id-start-time-index added successfully to {table_name}")

    except ClientError as e:
        logger.error(f"Failed to add index: {e}")
        if e.response["Error"]["Code"] == "ValidationException":
            logger.error("Index may already exist or attribute definitions may be missing")
        raise
    except Exception as e:
        logger.error(f"Failed to add index: {e}")
        raise


if __name__ == "__main__":
    add_missing_index()

