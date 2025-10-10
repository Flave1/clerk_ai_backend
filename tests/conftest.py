"""Pytest configuration and fixtures."""

import os
import pytest
import boto3
from moto import mock_dynamodb
from unittest.mock import Mock, patch
from datetime import datetime, timezone

# Set test environment variables
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["LIVEKIT_URL"] = "wss://test.livekit.cloud"
os.environ["LIVEKIT_API_KEY"] = "test_key"
os.environ["LIVEKIT_API_SECRET"] = "test_secret"
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["ANTHROPIC_API_KEY"] = "test_key"
os.environ["ELEVENLABS_API_KEY"] = "test_key"
os.environ["SLACK_BOT_TOKEN"] = "test_token"


@pytest.fixture
def dynamodb_client():
    """Provide a mock DynamoDB client."""
    with mock_dynamodb():
        client = boto3.client(
            "dynamodb",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        yield client


@pytest.fixture
def dynamodb_tables(dynamodb_client):
    """Create test DynamoDB tables."""
    tables = [
        {
            "TableName": "conversations",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "turns",
            "KeySchema": [{"AttributeName": "conversation_id", "KeyType": "HASH"}, {"AttributeName": "timestamp", "KeyType": "RANGE"}],
            "AttributeDefinitions": [
                {"AttributeName": "conversation_id", "AttributeType": "S"},
                {"AttributeName": "timestamp", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "actions",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "users",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "BillingMode": "PAY_PER_REQUEST",
        },
        {
            "TableName": "rooms",
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "BillingMode": "PAY_PER_REQUEST",
        },
    ]

    for table_config in tables:
        dynamodb_client.create_table(**table_config)

    yield tables

    # Cleanup
    for table_config in tables:
        try:
            dynamodb_client.delete_table(TableName=table_config["TableName"])
        except:
            pass


@pytest.fixture
def api_client():
    """Provide a test API client."""
    from fastapi.testclient import TestClient
    from services.api.main import app
    return TestClient(app)