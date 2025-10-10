"""
Mock CRM integration tool for demonstration.
"""
import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MockCRMTool:
    """Mock CRM integration tool."""

    def __init__(self):
        self.contacts = {}  # In-memory storage for demo
        logger.info("Mock CRM tool initialized")

    async def create_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new contact."""
        try:
            email = parameters.get("email")
            name = parameters.get("name", "")
            phone = parameters.get("phone", "")
            company = parameters.get("company", "")

            if not email:
                return {"success": False, "error": "email is required"}

            # Check if contact already exists
            if email in self.contacts:
                return {"success": False, "error": "Contact already exists"}

            # Create contact
            contact_id = f"contact_{len(self.contacts) + 1}"
            contact = {
                "id": contact_id,
                "email": email,
                "name": name,
                "phone": phone,
                "company": company,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "status": "active",
            }

            self.contacts[email] = contact

            logger.info(f"Created mock contact: {email}")

            return {
                "success": True,
                "result": {
                    "contact_id": contact_id,
                    "email": email,
                    "name": name,
                    "created_at": contact["created_at"],
                },
            }

        except Exception as e:
            logger.error(f"Failed to create contact: {e}")
            return {"success": False, "error": str(e)}

    async def update_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing contact."""
        try:
            email = parameters.get("email")
            if not email:
                return {"success": False, "error": "email is required"}

            # Check if contact exists
            if email not in self.contacts:
                return {"success": False, "error": "Contact not found"}

            contact = self.contacts[email]

            # Update fields
            if "name" in parameters:
                contact["name"] = parameters["name"]
            if "phone" in parameters:
                contact["phone"] = parameters["phone"]
            if "company" in parameters:
                contact["company"] = parameters["company"]
            if "status" in parameters:
                contact["status"] = parameters["status"]

            contact["updated_at"] = datetime.utcnow().isoformat()

            logger.info(f"Updated mock contact: {email}")

            return {
                "success": True,
                "result": {
                    "contact_id": contact["id"],
                    "email": email,
                    "updated_at": contact["updated_at"],
                },
            }

        except Exception as e:
            logger.error(f"Failed to update contact: {e}")
            return {"success": False, "error": str(e)}

    async def get_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get contact information."""
        try:
            email = parameters.get("email")
            if not email:
                return {"success": False, "error": "email is required"}

            if email not in self.contacts:
                return {"success": False, "error": "Contact not found"}

            contact = self.contacts[email]

            return {"success": True, "result": contact}

        except Exception as e:
            logger.error(f"Failed to get contact: {e}")
            return {"success": False, "error": str(e)}

    async def list_contacts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List all contacts."""
        try:
            limit = parameters.get("limit", 50)
            offset = parameters.get("offset", 0)

            # Get contacts
            contact_list = list(self.contacts.values())

            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_contacts = contact_list[start_idx:end_idx]

            return {
                "success": True,
                "result": {
                    "contacts": paginated_contacts,
                    "total": len(contact_list),
                    "limit": limit,
                    "offset": offset,
                },
            }

        except Exception as e:
            logger.error(f"Failed to list contacts: {e}")
            return {"success": False, "error": str(e)}

    async def delete_contact(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a contact."""
        try:
            email = parameters.get("email")
            if not email:
                return {"success": False, "error": "email is required"}

            if email not in self.contacts:
                return {"success": False, "error": "Contact not found"}

            contact = self.contacts[email]
            del self.contacts[email]

            logger.info(f"Deleted mock contact: {email}")

            return {
                "success": True,
                "result": {
                    "contact_id": contact["id"],
                    "email": email,
                    "deleted_at": datetime.utcnow().isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Failed to delete contact: {e}")
            return {"success": False, "error": str(e)}

    async def search_contacts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search contacts."""
        try:
            query = parameters.get("query", "")
            if not query:
                return {"success": False, "error": "query is required"}

            # Simple search implementation
            matching_contacts = []
            query_lower = query.lower()

            for contact in self.contacts.values():
                if (
                    query_lower in contact.get("name", "").lower()
                    or query_lower in contact.get("email", "").lower()
                    or query_lower in contact.get("company", "").lower()
                ):
                    matching_contacts.append(contact)

            return {
                "success": True,
                "result": {
                    "contacts": matching_contacts,
                    "count": len(matching_contacts),
                    "query": query,
                },
            }

        except Exception as e:
            logger.error(f"Failed to search contacts: {e}")
            return {"success": False, "error": str(e)}
