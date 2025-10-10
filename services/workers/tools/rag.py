"""
RAG (Retrieval Augmented Generation) tool for knowledge base search.
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RAGTool:
    """RAG tool for knowledge base search."""

    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        logger.info("RAG tool initialized")

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize a mock knowledge base."""
        return {
            "documents": [
                {
                    "id": "doc_1",
                    "title": "Company Overview",
                    "content": "Our company is a leading provider of AI-powered solutions. We specialize in real-time communication, natural language processing, and automation tools.",
                    "category": "company",
                    "tags": ["company", "overview", "ai"],
                },
                {
                    "id": "doc_2",
                    "title": "Meeting Room Booking",
                    "content": "To book a meeting room, you can use the calendar system or contact the receptionist. Meeting rooms are available from 9 AM to 6 PM on weekdays.",
                    "category": "facilities",
                    "tags": ["meeting", "booking", "calendar"],
                },
                {
                    "id": "doc_3",
                    "title": "Contact Information",
                    "content": "For general inquiries, contact info@company.com. For technical support, contact support@company.com. Emergency contact: +1-555-0123.",
                    "category": "contact",
                    "tags": ["contact", "email", "phone", "support"],
                },
                {
                    "id": "doc_4",
                    "title": "Office Hours",
                    "content": "Our office is open Monday through Friday from 9 AM to 6 PM. We are closed on weekends and major holidays.",
                    "category": "facilities",
                    "tags": ["hours", "office", "schedule"],
                },
                {
                    "id": "doc_5",
                    "title": "IT Support",
                    "content": "For IT support, please contact the IT department at it-support@company.com or call extension 1234. Common issues include password resets and software installation.",
                    "category": "support",
                    "tags": ["it", "support", "technical", "password"],
                },
            ]
        }

    async def search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search the knowledge base."""
        try:
            query = parameters.get("query", "")
            category = parameters.get("category")
            limit = parameters.get("limit", 5)

            if not query:
                return {"success": False, "error": "query is required"}

            # Simple keyword matching (in production, use vector similarity)
            query_lower = query.lower()
            matching_docs = []

            for doc in self.knowledge_base["documents"]:
                # Check if query matches content, title, or tags
                if (
                    query_lower in doc["content"].lower()
                    or query_lower in doc["title"].lower()
                    or any(query_lower in tag.lower() for tag in doc["tags"])
                ):
                    # Filter by category if specified
                    if category and doc["category"] != category:
                        continue

                    matching_docs.append(
                        {
                            "id": doc["id"],
                            "title": doc["title"],
                            "content": doc["content"],
                            "category": doc["category"],
                            "tags": doc["tags"],
                            "relevance_score": self._calculate_relevance(
                                query_lower, doc
                            ),
                        }
                    )

            # Sort by relevance score
            matching_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Limit results
            matching_docs = matching_docs[:limit]

            return {
                "success": True,
                "result": {
                    "query": query,
                    "documents": matching_docs,
                    "count": len(matching_docs),
                    "category": category,
                },
            }

        except Exception as e:
            logger.error(f"Failed to search knowledge base: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_relevance(self, query: str, doc: Dict[str, Any]) -> float:
        """Calculate relevance score for a document."""
        score = 0.0

        # Title match gets highest score
        if query in doc["title"].lower():
            score += 2.0

        # Content match gets medium score
        content_lower = doc["content"].lower()
        query_words = query.split()
        for word in query_words:
            if word in content_lower:
                score += 1.0

        # Tag match gets lower score
        for tag in doc["tags"]:
            if query in tag.lower():
                score += 0.5

        return score

    async def add_document(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add a document to the knowledge base."""
        try:
            title = parameters.get("title", "")
            content = parameters.get("content", "")
            category = parameters.get("category", "general")
            tags = parameters.get("tags", [])

            if not title or not content:
                return {"success": False, "error": "title and content are required"}

            # Generate document ID
            doc_id = f"doc_{len(self.knowledge_base['documents']) + 1}"

            # Create document
            document = {
                "id": doc_id,
                "title": title,
                "content": content,
                "category": category,
                "tags": tags,
            }

            self.knowledge_base["documents"].append(document)

            logger.info(f"Added document to knowledge base: {doc_id}")

            return {
                "success": True,
                "result": {"document_id": doc_id, "title": title, "category": category},
            }

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {"success": False, "error": str(e)}

    async def get_document(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific document by ID."""
        try:
            doc_id = parameters.get("doc_id")
            if not doc_id:
                return {"success": False, "error": "doc_id is required"}

            # Find document
            for doc in self.knowledge_base["documents"]:
                if doc["id"] == doc_id:
                    return {"success": True, "result": doc}

            return {"success": False, "error": "Document not found"}

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return {"success": False, "error": str(e)}

    async def list_categories(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """List all categories in the knowledge base."""
        try:
            categories = set()
            for doc in self.knowledge_base["documents"]:
                categories.add(doc["category"])

            return {
                "success": True,
                "result": {"categories": list(categories), "count": len(categories)},
            }

        except Exception as e:
            logger.error(f"Failed to list categories: {e}")
            return {"success": False, "error": str(e)}

    async def get_stats(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            total_docs = len(self.knowledge_base["documents"])

            # Count by category
            category_counts = {}
            for doc in self.knowledge_base["documents"]:
                category = doc["category"]
                category_counts[category] = category_counts.get(category, 0) + 1

            # Count total tags
            all_tags = set()
            for doc in self.knowledge_base["documents"]:
                all_tags.update(doc["tags"])

            return {
                "success": True,
                "result": {
                    "total_documents": total_docs,
                    "categories": category_counts,
                    "total_tags": len(all_tags),
                    "unique_tags": list(all_tags),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"success": False, "error": str(e)}
