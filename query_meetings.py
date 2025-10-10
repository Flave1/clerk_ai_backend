"""
Query and display all meetings from DynamoDB.
"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

from services.api.dao import DynamoDBDAO
from shared.schemas import Meeting


async def fetch_all_meetings(limit: int = 100) -> List[Meeting]:
    """Fetch all meetings from DynamoDB."""
    dao = DynamoDBDAO()
    await dao.initialize()
    
    meetings = await dao.get_meetings(limit=limit)
    return meetings


async def fetch_zoom_meetings(limit: int = 100) -> List[Meeting]:
    """Fetch only Zoom meetings from DynamoDB."""
    dao = DynamoDBDAO()
    await dao.initialize()
    
    # Get all meetings and filter for Zoom
    all_meetings = await dao.get_meetings(limit=limit)
    zoom_meetings = [m for m in all_meetings if m.platform.value == "zoom"]
    return zoom_meetings


async def display_meetings(meetings: List[Meeting]):
    """Display meetings in a formatted table."""
    if not meetings:
        print("❌ No meetings found in database")
        return
    
    print(f"\n{'='*100}")
    print(f"📅 MEETINGS IN DATABASE: {len(meetings)}")
    print(f"{'='*100}\n")
    
    for i, meeting in enumerate(meetings, 1):
        print(f"Meeting #{i}")
        print(f"  ID (Internal):     {meeting.id}")
        print(f"  ID (External):     {meeting.meeting_id_external}")
        print(f"  Platform:          {meeting.platform.value.upper()}")
        print(f"  Title:             {meeting.title}")
        print(f"  Join URL:          {meeting.meeting_url}")
        print(f"  Start Time:        {meeting.start_time}")
        print(f"  End Time:          {meeting.end_time}")
        print(f"  Status:            {meeting.status.value}")
        print(f"  Organizer:         {meeting.organizer_email}")
        
        if meeting.participants:
            print(f"  Participants:      {len(meeting.participants)}")
            for participant in meeting.participants:
                print(f"    - {participant.email} ({participant.response_status})")
        else:
            print(f"  Participants:      None")
        
        print(f"  Created:           {meeting.created_at}")
        print()
    
    print(f"{'='*100}\n")


async def get_meeting_by_id(meeting_id: str) -> Meeting:
    """Get a specific meeting by ID."""
    dao = DynamoDBDAO()
    await dao.initialize()
    
    meeting = await dao.get_meeting(meeting_id)
    return meeting


async def get_recent_meetings(limit: int = 10) -> List[Meeting]:
    """Get most recent meetings."""
    dao = DynamoDBDAO()
    await dao.initialize()
    
    meetings = await dao.get_meetings(limit=100)
    # Sort by created_at descending
    sorted_meetings = sorted(meetings, key=lambda m: m.created_at, reverse=True)
    return sorted_meetings[:limit]


async def export_meetings_to_json(filename: str = "meetings_export.json"):
    """Export all meetings to JSON file."""
    meetings = await fetch_all_meetings()
    
    meetings_data = []
    for meeting in meetings:
        meetings_data.append({
            "id": str(meeting.id),
            "external_id": meeting.meeting_id_external,
            "platform": meeting.platform.value,
            "title": meeting.title,
            "url": meeting.meeting_url,
            "start_time": meeting.start_time.isoformat(),
            "end_time": meeting.end_time.isoformat(),
            "status": meeting.status.value,
            "organizer": meeting.organizer_email,
            "participants": [
                {"email": p.email, "name": p.name, "status": p.response_status}
                for p in (meeting.participants or [])
            ],
            "created_at": meeting.created_at.isoformat()
        })
    
    with open(filename, 'w') as f:
        json.dump(meetings_data, f, indent=2)
    
    print(f"✅ Exported {len(meetings_data)} meetings to {filename}")


async def get_meeting_stats():
    """Get statistics about meetings."""
    meetings = await fetch_all_meetings()
    
    if not meetings:
        print("❌ No meetings found")
        return
    
    # Calculate stats
    total = len(meetings)
    zoom_count = len([m for m in meetings if m.platform.value == "zoom"])
    google_count = len([m for m in meetings if m.platform.value == "google_meet"])
    
    status_counts = {}
    for meeting in meetings:
        status = meeting.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print(f"\n{'='*80}")
    print(f"📊 MEETING STATISTICS")
    print(f"{'='*80}")
    print(f"\n  Total Meetings:        {total}")
    print(f"  Zoom Meetings:         {zoom_count}")
    print(f"  Google Meet Meetings:  {google_count}")
    print(f"\n  Status Breakdown:")
    for status, count in status_counts.items():
        print(f"    {status.capitalize()}: {count}")
    print(f"\n{'='*80}\n")


async def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "all":
            meetings = await fetch_all_meetings()
            await display_meetings(meetings)
        
        elif command == "zoom":
            meetings = await fetch_zoom_meetings()
            await display_meetings(meetings)
        
        elif command == "recent":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            meetings = await get_recent_meetings(limit)
            await display_meetings(meetings)
        
        elif command == "stats":
            await get_meeting_stats()
        
        elif command == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "meetings_export.json"
            await export_meetings_to_json(filename)
        
        elif command == "get":
            if len(sys.argv) < 3:
                print("❌ Please provide meeting ID")
                return
            meeting_id = sys.argv[2]
            meeting = await get_meeting_by_id(meeting_id)
            if meeting:
                await display_meetings([meeting])
            else:
                print(f"❌ Meeting {meeting_id} not found")
        
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()
    
    else:
        # Default: show stats and recent meetings
        await get_meeting_stats()
        print("\n📅 Recent Meetings (Last 5):\n")
        meetings = await get_recent_meetings(5)
        await display_meetings(meetings)


def print_usage():
    """Print usage instructions."""
    print("""
Usage: python query_meetings.py [command] [options]

Commands:
  all              - List all meetings
  zoom             - List only Zoom meetings
  recent [N]       - List N most recent meetings (default: 10)
  stats            - Show meeting statistics
  export [file]    - Export all meetings to JSON file
  get <id>         - Get specific meeting by ID
  
Examples:
  python query_meetings.py all
  python query_meetings.py zoom
  python query_meetings.py recent 5
  python query_meetings.py stats
  python query_meetings.py export meetings.json
  python query_meetings.py get 99442d5f-3b63-4e44-86e3-b25632a8ba16
""")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
