# Fireflies.ai Integration Guide

## Overview

The Clerk meeting agent now supports integration with Fireflies.ai to automatically join, record, and transcribe meetings across platforms (Zoom, Google Meet, Microsoft Teams).

Fireflies.ai provides:
- ✅ Automatic bot joining to meetings
- 🎙️ Audio/video recording
- 📝 Real-time transcription with speaker identification
- 📊 AI-generated meeting summaries
- 🎯 Action item extraction
- 🔍 Keyword detection and meeting analytics

## Benefits

Using Fireflies instead of custom SDK integration provides:
1. **Faster deployment** - No need to set up complex Zoom/Teams SDKs
2. **Better reliability** - Fireflies handles all edge cases for meeting platforms
3. **Enhanced features** - Get professional transcription, summaries, and analytics
4. **Multi-platform support** - Works with Zoom, Google Meet, Teams, and more with one API
5. **Reduced maintenance** - Fireflies handles platform API changes and updates

## Setup

### 1. Get Fireflies API Key

1. Sign up at [Fireflies.ai](https://fireflies.ai)
2. Navigate to **Settings** → **Integrations** → **Fireflies API**
3. Generate an API key
4. Copy your API key

### 2. Configure Environment Variables

Add your Fireflies API key to your environment configuration:

```bash
# .env file
FIREFLIES_API_KEY=your-fireflies-api-key-here
```

### 3. Verify Configuration

The Fireflies API key is automatically loaded via the `Settings` class in `shared/config.py`:

```python
from shared.config import get_settings
settings = get_settings()
print(settings.fireflies_api_key)  # Should print your API key
```

## Usage

### Joining a Meeting

The `ZoomClientWrapper` (and other meeting clients) now use Fireflies to join meetings:

```python
from services.meeting_agent.zoom_client import ZoomClientWrapper
from shared.schemas import Meeting

# Initialize client
client = ZoomClientWrapper()
await client.initialize()

# Create meeting object
meeting = Meeting(
    id="...",
    platform=MeetingPlatform.ZOOM,
    meeting_url="https://zoom.us/j/123456789?pwd=abc123",
    title="Project Sync",
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(hours=1)
)

# Join via Fireflies
response = await client.join_meeting(meeting)

if response.success:
    print(f"✅ Bot joined meeting!")
    print(f"📝 Transcript ID: {response.metadata['fireflies_transcript_id']}")
else:
    print(f"❌ Failed: {response.error_message}")
```

### Retrieving Transcripts

After the meeting ends, retrieve the full transcript and AI-generated summary:

```python
# Get transcript from Fireflies
transcript_id = response.metadata['fireflies_transcript_id']
transcript_data = await client.get_fireflies_transcript(transcript_id)

# Access transcript content
full_text = transcript_data['transcript_text']
sentences = transcript_data['sentences']  # With speaker identification

# Access AI-generated summary
summary = transcript_data['summary']
overview = summary['overview']
action_items = summary['action_items']
keywords = summary['keywords']

print(f"Meeting Overview: {overview}")
print(f"Action Items: {action_items}")
```

### Syncing to Database

Automatically sync Fireflies transcript data to your database:

```python
# Sync transcript and summary to database
success = await client.sync_fireflies_transcript_to_db(
    meeting_id=meeting.id,
    transcript_id=transcript_id
)

if success:
    print("✅ Transcript and summary saved to database")
```

## API Reference

### `_join_via_fireflies_api(meeting: Meeting) -> Dict`

Sends a request to Fireflies to add their bot to the meeting.

**Parameters:**
- `meeting`: Meeting object with details (URL, title, attendees, etc.)

**Returns:**
```python
{
    'success': True,
    'transcript_id': 'abc123...',
    'meeting_id': 'abc123...',
    'message': 'Bot added successfully'
}
```

### `get_fireflies_transcript(transcript_id: str) -> Dict`

Retrieves complete transcript data from Fireflies.

**Parameters:**
- `transcript_id`: Fireflies transcript ID

**Returns:**
```python
{
    'id': 'abc123...',
    'title': 'Project Sync',
    'date': '2025-10-10T14:00:00Z',
    'duration': 3600,
    'transcript_text': 'Full transcript...',
    'sentences': [
        {
            'speaker_name': 'John Doe',
            'text': 'Hello everyone',
            'start_time': 0.0,
            'end_time': 2.5
        },
        ...
    ],
    'summary': {
        'overview': 'Meeting summary...',
        'action_items': ['Action 1', 'Action 2'],
        'keywords': ['project', 'deadline', 'budget'],
        'outline': [...]
    },
    'participants': [
        {'name': 'John Doe', 'email': 'john@example.com'},
        ...
    ]
}
```

### `sync_fireflies_transcript_to_db(meeting_id: str, transcript_id: str) -> bool`

Retrieves transcript from Fireflies and stores in database.

**Parameters:**
- `meeting_id`: Internal meeting UUID
- `transcript_id`: Fireflies transcript ID

**Returns:**
- `True` if successful, `False` otherwise

## Fireflies GraphQL API

Fireflies uses a GraphQL API. Here's the schema used:

### Create Transcript (Join Meeting)

```graphql
mutation($input: TranscriptInput!) {
    createTranscript(input: $input) {
        success
        message
        transcript {
            id
            title
            meeting_url
            date
        }
    }
}
```

**Variables:**
```json
{
    "input": {
        "title": "Meeting Title",
        "meeting_url": "https://zoom.us/j/123456789",
        "start_time": "2025-10-10T14:00:00Z",
        "attendees": [
            {"email": "user@example.com", "name": "User Name"}
        ],
        "auto_join": true,
        "record_audio": true,
        "record_video": false
    }
}
```

### Get Transcript

```graphql
query($transcriptId: String!) {
    transcript(id: $transcriptId) {
        id
        title
        date
        duration
        transcript_text
        sentences {
            speaker_name
            text
            start_time
            end_time
        }
        summary {
            overview
            action_items
            keywords
            outline
        }
    }
}
```

## Troubleshooting

### API Key Not Configured

**Error:** `Fireflies API key not configured`

**Solution:** Ensure `FIREFLIES_API_KEY` is set in your `.env` file.

### Bot Failed to Join

**Error:** `Fireflies API error: Bot failed to join meeting`

**Possible causes:**
1. Invalid meeting URL
2. Meeting hasn't started yet
3. Meeting requires registration
4. Fireflies bot was blocked by host

**Solution:** 
- Verify meeting URL is correct and publicly accessible
- Ensure meeting is scheduled or currently in progress
- Check that the meeting allows bots/external participants

### Transcript Not Available

**Error:** `No transcript data found for ID`

**Possible causes:**
1. Meeting hasn't ended yet
2. Transcript is still being processed
3. No audio was captured

**Solution:**
- Wait 2-5 minutes after meeting ends for processing
- Check Fireflies dashboard to see transcript status
- Poll the API periodically until transcript is ready

## Best Practices

1. **Store transcript IDs**: Always save the `fireflies_transcript_id` returned from `join_meeting()` for later retrieval

2. **Handle delays**: Transcripts may take a few minutes to process after the meeting ends. Implement polling or webhook handling.

3. **Error handling**: Always check the `success` field in API responses and handle errors gracefully

4. **Rate limiting**: Fireflies API has rate limits. Implement exponential backoff for retries.

5. **Webhook integration** (Advanced): Configure Fireflies webhooks to get notified when transcripts are ready instead of polling.

## Migration from Custom SDK

If migrating from custom Zoom/Teams SDK integration:

1. ✅ Replace SDK initialization with Fireflies API calls
2. ✅ Remove custom audio capture code (Fireflies handles this)
3. ✅ Remove custom transcription pipeline (Fireflies provides this)
4. ✅ Update database schema to store `fireflies_transcript_id`
5. ✅ Update transcript retrieval to use Fireflies API
6. ✅ Test with various meeting platforms

## Additional Resources

- [Fireflies API Documentation](https://docs.fireflies.ai/)
- [Fireflies Dashboard](https://app.fireflies.ai/)
- [Fireflies GraphQL Playground](https://api.fireflies.ai/graphql)
- [Support](https://help.fireflies.ai/)

## Example: Complete Workflow

```python
import asyncio
from services.meeting_agent.zoom_client import ZoomClientWrapper
from shared.schemas import Meeting, MeetingPlatform
from datetime import datetime, timedelta

async def run_meeting_with_fireflies():
    """Complete example of joining a meeting with Fireflies."""
    
    # Initialize client
    client = ZoomClientWrapper()
    await client.initialize()
    
    # Create meeting
    meeting = Meeting(
        id="550e8400-e29b-41d4-a716-446655440000",
        platform=MeetingPlatform.ZOOM,
        meeting_url="https://zoom.us/j/123456789?pwd=abc123",
        title="Weekly Team Sync",
        start_time=datetime.now(),
        end_time=datetime.now() + timedelta(hours=1)
    )
    
    # Join meeting via Fireflies
    print("🤖 Joining meeting...")
    response = await client.join_meeting(meeting)
    
    if not response.success:
        print(f"❌ Failed to join: {response.error_message}")
        return
    
    transcript_id = response.metadata['fireflies_transcript_id']
    print(f"✅ Bot joined successfully!")
    print(f"📝 Transcript ID: {transcript_id}")
    
    # Wait for meeting to end (in production, use webhooks or scheduled jobs)
    print("⏳ Waiting for meeting to end and transcript to process...")
    await asyncio.sleep(300)  # Wait 5 minutes (adjust as needed)
    
    # Retrieve transcript
    print("📥 Retrieving transcript from Fireflies...")
    transcript_data = await client.get_fireflies_transcript(transcript_id)
    
    if transcript_data:
        print(f"✅ Transcript retrieved!")
        print(f"📊 Duration: {transcript_data['duration']} seconds")
        print(f"👥 Participants: {len(transcript_data['participants'])}")
        print(f"\n📝 Summary Overview:")
        print(transcript_data['summary']['overview'])
        print(f"\n🎯 Action Items:")
        for item in transcript_data['summary']['action_items']:
            print(f"  - {item}")
        
        # Sync to database
        print("\n💾 Syncing to database...")
        success = await client.sync_fireflies_transcript_to_db(
            meeting_id=meeting.id,
            transcript_id=transcript_id
        )
        print(f"{'✅' if success else '❌'} Database sync completed")
    else:
        print("❌ Failed to retrieve transcript")
    
    # Cleanup
    await client.cleanup()
    print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(run_meeting_with_fireflies())
```

---

**Note**: This integration replaces the need for custom Zoom Meeting SDK or Microsoft Teams SDK implementations, significantly simplifying the meeting bot architecture.

