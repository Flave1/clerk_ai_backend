#!/usr/bin/env python3
"""
Script to create DynamoDB tables for the AI Receptionist system.
"""
import asyncio
import logging
import os
import sys

# Add the parent directory to the path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from botocore.exceptions import ClientError

from shared.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


def create_dynamodb_tables():
    """Create all required DynamoDB tables."""
    try:
        # Initialize DynamoDB client
        if settings.aws_access_key_id:
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            # Use local DynamoDB for development
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url="http://localhost:8001",
            )

        # Table configurations
        tables = [
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.conversations_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "started_at", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "user-id-index",
                    "KeySchema": [
                        {"AttributeName": "user_id", "KeyType": "HASH"},
                        {"AttributeName": "started_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.turns_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "conversation_id", "AttributeType": "S"},
                {"AttributeName": "turn_number", "AttributeType": "N"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "conversation-id-index",
                    "KeySchema": [
                        {"AttributeName": "conversation_id", "KeyType": "HASH"},
                        {"AttributeName": "turn_number", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.actions_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "conversation_id", "AttributeType": "S"},
                {"AttributeName": "action_type", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "conversation-id-index",
                    "KeySchema": [
                        {"AttributeName": "conversation_id", "KeyType": "HASH"},
                        {"AttributeName": "action_type", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "status-index",
                    "KeySchema": [
                        {"AttributeName": "status", "KeyType": "HASH"},
                        {"AttributeName": "action_type", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.users_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "email", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "email-index",
                    "KeySchema": [{"AttributeName": "email", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}rooms",
            "KeySchema": [{"AttributeName": "room_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "room_id", "AttributeType": "S"},
                {"AttributeName": "is_active", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "is-active-index",
                    "KeySchema": [
                        {"AttributeName": "is_active", "KeyType": "HASH"},
                        {"AttributeName": "room_id", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.meetings_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "platform", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
                {"AttributeName": "start_time", "AttributeType": "S"},
                {"AttributeName": "ai_email", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "platform-status-index",
                    "KeySchema": [
                        {"AttributeName": "platform", "KeyType": "HASH"},
                        {"AttributeName": "status", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
                {
                    "IndexName": "ai-email-start-time-index",
                    "KeySchema": [
                        {"AttributeName": "ai_email", "KeyType": "HASH"},
                        {"AttributeName": "start_time", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.meeting_summaries_table}",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "meeting_id", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "meeting-id-index",
                    "KeySchema": [
                        {"AttributeName": "meeting_id", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.meeting_transcriptions_table}",
            "KeySchema": [{"AttributeName": "chunk_id", "KeyType": "HASH"}],
            "AttributeDefinitions": [
                {"AttributeName": "chunk_id", "AttributeType": "S"},
                {"AttributeName": "meeting_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "meeting-id-timestamp-index",
                    "KeySchema": [
                        {"AttributeName": "meeting_id", "KeyType": "HASH"},
                        {"AttributeName": "timestamp", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
        {
            "TableName": f"{settings.dynamodb_table_prefix}{settings.meeting_notifications_table}",
            "KeySchema": [{"AttributeName": "meeting_id", "KeyType": "HASH"}, {"AttributeName": "notification_type", "KeyType": "RANGE"}],
            "AttributeDefinitions": [
                {"AttributeName": "meeting_id", "AttributeType": "S"},
                {"AttributeName": "notification_type", "AttributeType": "S"},
                {"AttributeName": "sent_at", "AttributeType": "S"},
                {"AttributeName": "delivery_status", "AttributeType": "S"},
            ],
            "GlobalSecondaryIndexes": [
                {
                    "IndexName": "delivery-status-sent-at-index",
                    "KeySchema": [
                        {"AttributeName": "delivery_status", "KeyType": "HASH"},
                        {"AttributeName": "sent_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            "ProvisionedThroughput": {
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        },
]


        # Create tables
        for table_config in tables:
            table_name = table_config["TableName"]

            try:
                # Check if table exists
                table = dynamodb.Table(table_name)
                table.load()
                logger.info(f"Table {table_name} already exists")
                continue

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    # Table doesn't exist, create it
                    pass
                else:
                    raise

            try:
                logger.info(f"Creating table: {table_name}")

                table = dynamodb.create_table(**table_config)

                # Wait for table to be created
                logger.info(f"Waiting for table {table_name} to be created...")
                table.wait_until_exists()

                logger.info(f"Table {table_name} created successfully")

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceInUseException":
                    logger.info(f"Table {table_name} already exists")
                else:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    raise

        logger.info("All tables created successfully!")

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


def delete_dynamodb_tables():
    """Delete all DynamoDB tables (for development/testing)."""
    try:
        # Initialize DynamoDB client
        if settings.aws_access_key_id:
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
        else:
            # Use local DynamoDB for development
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url="http://localhost:8001",
            )

        # List of tables to delete
        table_names = [
            f"{settings.dynamodb_table_prefix}{settings.conversations_table}",
            f"{settings.dynamodb_table_prefix}{settings.turns_table}",
            f"{settings.dynamodb_table_prefix}{settings.actions_table}",
            f"{settings.dynamodb_table_prefix}{settings.users_table}",
            f"{settings.dynamodb_table_prefix}rooms",
            f"{settings.dynamodb_table_prefix}{settings.meetings_table}",
            f"{settings.dynamodb_table_prefix}{settings.meeting_summaries_table}",
            f"{settings.dynamodb_table_prefix}{settings.meeting_transcriptions_table}",
            f"{settings.dynamodb_table_prefix}{settings.meeting_notifications_table}",
        ]

        for table_name in table_names:
            try:
                table = dynamodb.Table(table_name)
                table.delete()
                logger.info(f"Deleting table: {table_name}")

                # Wait for table to be deleted
                table.wait_until_not_exists()
                logger.info(f"Table {table_name} deleted successfully")

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    logger.info(f"Table {table_name} does not exist")
                else:
                    logger.error(f"Failed to delete table {table_name}: {e}")

        logger.info("All tables deleted successfully!")

    except Exception as e:
        logger.error(f"Failed to delete tables: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage DynamoDB tables")
    parser.add_argument(
        "action", choices=["create", "delete"], help="Action to perform"
    )

    args = parser.parse_args()

    if args.action == "create":
        create_dynamodb_tables()
    elif args.action == "delete":
        confirm = input("Are you sure you want to delete all tables? (yes/no): ")
        if confirm.lower() == "yes":
            delete_dynamodb_tables()
        else:
            logger.info("Operation cancelled")
