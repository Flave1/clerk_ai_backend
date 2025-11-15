#!/usr/bin/env python3
"""
Script to seed integrations for a user.
Creates UserIntegration records in DISCONNECTED status with realistic data.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

# Add the parent directory to the path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import get_settings
from shared.schemas import UserIntegration, IntegrationStatus
from services.api.dao import DynamoDBDAO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Integration configurations with realistic data
INTEGRATIONS_TO_SEED = [
    {
        "integration_id": "slack",
        "name": "Slack",
        "scope": "channels:read,channels:write,chat:write,chat:write.public,files:read,files:write,users:read,users:read.email",
        "metadata": {
            "workspace_name": "Acme Corporation",
            "workspace_id": "T01234567",
            "team_name": "Engineering",
            "app_id": "A01234567",
            "bot_user_id": "U01234567",
            "workspace_url": "acme-corp.slack.com",
            "plan": "Business+",
        }
    },
    {
        "integration_id": "google_workspace",
        "name": "Google Workspace",
        "scope": "https://www.googleapis.com/auth/calendar,https://www.googleapis.com/auth/gmail.send,https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/documents.readonly",
        "metadata": {
            "account_email": "john.doe@acmecorp.com",
            "account_type": "user",
            "domain": "acmecorp.com",
            "services_enabled": ["calendar", "gmail", "drive", "docs", "sheets", "slides"],
            "primary_calendar_id": "primary",
            "calendar_timezone": "America/New_York",
            "workspace_plan": "Business Standard",
        }
    },
    {
        "integration_id": "microsoft_365",
        "name": "Microsoft 365",
        "scope": "https://graph.microsoft.com/Calendars.ReadWrite,https://graph.microsoft.com/Mail.ReadWrite,https://graph.microsoft.com/Files.ReadWrite,https://graph.microsoft.com/Sites.ReadWrite.All,offline_access",
        "metadata": {
            "account_email": "john.doe@acmecorp.com",
            "account_type": "user",
            "tenant_id": "12345678-1234-1234-1234-123456789012",
            "tenant_name": "Acme Corporation",
            "services_enabled": ["outlook", "teams", "onedrive", "sharepoint", "excel", "word"],
            "subscription": "Microsoft 365 Business Premium",
            "timezone": "Eastern Standard Time",
        }
    },
    {
        "integration_id": "zoom",
        "name": "Zoom",
        "scope": "meeting:write,meeting:read,user:read,recording:read",
        "metadata": {
            "account_email": "john.doe@acmecorp.com",
            "account_type": "pro",
            "account_id": "abc123xyz",
            "meeting_capacity": 100,
            "webinar_enabled": True,
            "recording_enabled": True,
            "cloud_recording_storage": "50GB",
            "phone_enabled": False,
            "account_name": "Acme Corporation",
        }
    },
    {
        "integration_id": "hubspot",
        "name": "HubSpot",
        "scope": "contacts,deals,companies,timeline,content",
        "metadata": {
            "portal_id": "12345678",
            "account_name": "Acme Corporation",
            "account_type": "professional",
            "api_key_set": True,
            "crm_enabled": True,
            "marketing_hub": True,
            "sales_hub": True,
            "service_hub": False,
            "contacts_count": 1250,
            "deals_pipeline": "Sales Pipeline",
        }
    },
    {
        "integration_id": "jira",
        "name": "Jira",
        "scope": "read:board,write:issue,read:issue,read:project,write:project,manage:sprint",
        "metadata": {
            "site_url": "https://acmecorp.atlassian.net",
            "organization": "Acme Corporation",
            "project_keys": ["ENG", "OPS", "PLAT"],
            "default_project": "ENG",
            "product": "Jira Software Cloud",
            "plan": "Premium",
            "issue_types": ["Story", "Bug", "Task", "Epic"],
            "boards": [
                {"name": "Engineering Kanban", "type": "kanban"},
                {"name": "Platform Sprint Board", "type": "scrum"}
            ],
            "timezone": "America/New_York"
        }
    },
]


async def seed_integrations(user_id: str):
    """Seed integrations for a user."""
    try:
        # Initialize DAO
        dao = DynamoDBDAO()
        await dao.initialize()
        
        user_uuid = UUID(user_id)
        logger.info(f"Seeding integrations for user: {user_id}")
        
        created_count = 0
        skipped_count = 0
        
        for integration_config in INTEGRATIONS_TO_SEED:
            integration_id = integration_config["integration_id"]
            
            try:
                # Check if integration already exists
                existing = await dao.get_user_integration(str(user_uuid), integration_id)
                
                now = datetime.now(timezone.utc)
                # Set connected_at to a few days ago to make it look real
                connected_at = now - timedelta(days=7)
                # Set last_used_at to a few hours ago
                last_used_at = now - timedelta(hours=3)
                
                if existing:
                    # Update existing integration to CONNECTED status
                    existing.status = IntegrationStatus.CONNECTED
                    existing.access_token = "mock_access_token_placeholder"
                    existing.refresh_token = "mock_refresh_token_placeholder"
                    existing.token_type = "Bearer"
                    existing.expires_at = now + timedelta(hours=1)
                    existing.scope = integration_config.get("scope", "read write")
                    existing.connected_at = connected_at
                    existing.last_used_at = last_used_at
                    existing.last_refreshed_at = now - timedelta(days=1)
                    existing.error_message = None
                    existing.metadata = integration_config["metadata"]
                    existing.updated_at = now
                    
                    await dao.update_user_integration(existing)
                    logger.info(f"✓ Updated integration '{integration_config['name']}' ({integration_id}) to CONNECTED status for user {user_id}")
                    created_count += 1
                else:
                    # Create new integration in CONNECTED status with realistic connection data
                    user_integration = UserIntegration(
                        id=uuid4(),
                        user_id=user_uuid,
                        integration_id=integration_id,
                        status=IntegrationStatus.CONNECTED,
                        access_token="mock_access_token_placeholder",  # Placeholder - will be replaced with real token when connecting
                        refresh_token="mock_refresh_token_placeholder",  # Placeholder
                        token_type="Bearer",
                        expires_at=now + timedelta(hours=1),  # Token expires in 1 hour
                        scope=integration_config.get("scope", "read write"),  # Default scopes
                        connected_at=connected_at,
                        last_used_at=last_used_at,
                        last_refreshed_at=now - timedelta(days=1),  # Refreshed yesterday
                        error_message=None,
                        metadata=integration_config["metadata"],
                        created_at=connected_at,  # Created when connected
                        updated_at=now,
                    )
                    
                    # Save to database
                    await dao.create_user_integration(user_integration)
                    logger.info(f"✓ Created integration '{integration_config['name']}' ({integration_id}) for user {user_id}")
                    created_count += 1
                
            except Exception as e:
                logger.error(f"Failed to create integration '{integration_id}': {e}")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Seeding complete!")
        logger.info(f"  Processed: {created_count} integrations (created or updated to CONNECTED)")
        logger.info(f"  Skipped: {skipped_count} integrations")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Failed to seed integrations: {e}", exc_info=True)
        raise


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed integrations for a user")
    parser.add_argument(
        "--user-id",
        type=str,
        default="ab3b6eef-4e92-4c7e-b270-32c8405c6b67",
        help="User ID to seed integrations for",
    )
    
    args = parser.parse_args()
    
    # Validate user ID
    try:
        UUID(args.user_id)
    except ValueError:
        logger.error(f"Invalid user ID format: {args.user_id}")
        sys.exit(1)
    
    await seed_integrations(args.user_id)


if __name__ == "__main__":
    asyncio.run(main())

