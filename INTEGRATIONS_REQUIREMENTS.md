# Integrations System - Backend Requirements

## Overview
This document outlines what's needed to connect third-party apps to Aurray. Each integration requires OAuth 2.0 authentication (with some platform-specific variations).

---

## 1. Database Schema Requirements

### New Table: `user_integrations`
Store user's connected integrations and OAuth tokens.

**Schema:**
```python
{
    "id": UUID (Primary Key),
    "user_id": UUID (Partition Key - GSI),
    "integration_id": str (e.g., "google_workspace", "slack"),
    "status": str ("connected", "disconnected", "expired", "error"),
    
    # OAuth Tokens (encrypted at rest)
    "access_token": str (encrypted),
    "refresh_token": str (encrypted, optional),
    "token_type": str (usually "Bearer"),
    "expires_at": datetime (optional),
    "scope": str (comma-separated scopes),
    
    # Integration Metadata
    "connected_at": datetime,
    "last_used_at": datetime (optional),
    "last_refreshed_at": datetime (optional),
    "error_message": str (optional),
    
    # Platform-specific data (JSON)
    "metadata": dict (e.g., user email, workspace ID, etc.)
}
```

**Indexes:**
- GSI: `user_id` (to get all integrations for a user)
- GSI: `integration_id` (to get all users with a specific integration)

---

## 2. Integration Configuration Schema

### New Table: `integrations` (or config file)
Store available integrations and their OAuth configuration.

**Schema:**
```python
{
    "id": str (e.g., "google_workspace"),
    "name": str,
    "description": str,
    "category": str,
    "image_url": str,
    
    # OAuth Configuration
    "oauth_version": str ("2.0", "1.0"),
    "authorization_url": str,
    "token_url": str,
    "refresh_url": str (optional),
    "revoke_url": str (optional),
    
    # Required Scopes
    "default_scopes": list[str],
    
    # Client Credentials (from environment or secure storage)
    "client_id": str (from env),
    "client_secret": str (from env, encrypted),
    
    # Platform-specific settings
    "redirect_uri": str,
    "response_type": str (usually "code"),
    "grant_type": str (usually "authorization_code"),
    
    # Additional OAuth params
    "extra_params": dict (optional)
}
```

---

## 3. Platform-Specific OAuth Requirements

### 3.1 Google Workspace (Calendar, Gmail, Drive, Docs)
**OAuth Type:** OAuth 2.0
**Requirements:**
- Google Cloud Project
- OAuth 2.0 Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/google_workspace/callback`
- Scopes:
  - Calendar: `https://www.googleapis.com/auth/calendar`
  - Gmail: `https://www.googleapis.com/auth/gmail.send`
  - Drive: `https://www.googleapis.com/auth/drive.readonly`
  - Docs: `https://www.googleapis.com/auth/documents`

**Special Notes:**
- Uses `access_type=offline` to get refresh tokens
- Token refresh uses Google's token endpoint
- Can use service account for server-to-server (alternative)

---

### 3.2 Microsoft 365 (Outlook, Teams, OneDrive, SharePoint)
**OAuth Type:** OAuth 2.0 (Microsoft Identity Platform)
**Requirements:**
- Azure AD App Registration
- Client ID & Secret
- Tenant ID (or "common" for multi-tenant)
- Redirect URI: `https://yourdomain.com/api/v1/integrations/microsoft_365/callback`
- Scopes:
  - Outlook: `https://graph.microsoft.com/Mail.ReadWrite`
  - Calendar: `https://graph.microsoft.com/Calendars.ReadWrite`
  - Teams: `https://graph.microsoft.com/Team.ReadBasic.All`
  - OneDrive: `https://graph.microsoft.com/Files.ReadWrite`

**Special Notes:**
- Uses Microsoft Graph API
- Token refresh uses `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`
- Requires admin consent for some scopes

---

### 3.3 Slack
**OAuth Type:** OAuth 2.0
**Requirements:**
- Slack App (created at api.slack.com)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/slack/callback`
- Scopes (Bot Token):
  - `chat:write`
  - `channels:read`
  - `users:read`
  - `files:write`

**Special Notes:**
- Can use Bot Token or User Token
- Bot tokens don't expire (unless revoked)
- User tokens can expire and need refresh

---

### 3.4 Notion
**OAuth Type:** OAuth 2.0
**Requirements:**
- Notion Integration (created at notion.so/my-integrations)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/notion/callback`
- Scopes: Notion uses "capabilities" instead of scopes
  - `read`
  - `update`
  - `insert`

**Special Notes:**
- Uses Notion API v1
- Access tokens don't expire (unless revoked)
- Requires workspace connection

---

### 3.5 Zoom
**OAuth Type:** OAuth 2.0
**Requirements:**
- Zoom App (created at marketplace.zoom.us)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/zoom/callback`
- Scopes:
  - `meeting:write`
  - `user:read`
  - `recording:read`

**Special Notes:**
- Can use Server-to-Server OAuth (no user interaction)
- Or User OAuth (requires user consent)
- Token refresh available

---

### 3.6 HubSpot
**OAuth Type:** OAuth 2.0
**Requirements:**
- HubSpot App (created at developers.hubspot.com)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/hubspot/callback`
- Scopes:
  - `contacts`
  - `deals`
  - `timeline`

**Special Notes:**
- Uses HubSpot API v3
- Token refresh available
- Requires portal connection

---

### 3.7 Salesforce
**OAuth Type:** OAuth 2.0
**Requirements:**
- Salesforce Connected App
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/salesforce/callback`
- Scopes:
  - `api`
  - `refresh_token`
  - `full`

**Special Notes:**
- Uses Salesforce REST API
- Requires instance URL (e.g., `https://yourinstance.salesforce.com`)
- Token refresh available

---

### 3.8 GitHub
**OAuth Type:** OAuth 2.0
**Requirements:**
- GitHub OAuth App (created at github.com/settings/developers)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/github/callback`
- Scopes:
  - `repo`
  - `issues:write`
  - `pull_requests:write`

**Special Notes:**
- Uses GitHub REST API
- Token refresh not available (tokens don't expire unless revoked)
- Can use GitHub Apps (more advanced)

---

### 3.9 Jira
**OAuth Type:** OAuth 2.0 (3LO - 3-legged OAuth)
**Requirements:**
- Jira OAuth App (created in Jira settings)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/jira/callback`
- Scopes:
  - `read:jira-work`
  - `write:jira-work`

**Special Notes:**
- Requires Jira instance URL
- Uses Jira REST API v3
- Token refresh available

---

### 3.10 Asana
**OAuth Type:** OAuth 2.0
**Requirements:**
- Asana App (created at developers.asana.com)
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/asana/callback`
- Scopes: Asana doesn't use scopes (all or nothing)

**Special Notes:**
- Uses Asana API v1
- Token refresh available
- Requires workspace connection

---

### 3.11 Trello
**OAuth Type:** OAuth 1.0a (legacy, but still supported)
**Requirements:**
- Trello API Key & Secret (from trello.com/app-key)
- Redirect URI: `https://yourdomain.com/api/v1/integrations/trello/callback`
- Scopes: `read`, `write`, `account`

**Special Notes:**
- Uses OAuth 1.0a (different from OAuth 2.0)
- Tokens don't expire
- Uses Trello REST API v1

---

### 3.12 Stripe
**OAuth Type:** OAuth 2.0
**Requirements:**
- Stripe Connect App
- Client ID & Secret
- Redirect URI: `https://yourdomain.com/api/v1/integrations/stripe/callback`
- Scopes:
  - `read_only`
  - `read_write`

**Special Notes:**
- Uses Stripe Connect (for marketplace apps)
- Or Stripe API keys (simpler, but less secure)
- Token refresh available

---

## 4. Backend API Endpoints Needed

### 4.1 List Available Integrations
```
GET /api/v1/integrations
Response: List of all available integrations with metadata
```

### 4.2 Get Integration Details
```
GET /api/v1/integrations/{integration_id}
Response: Integration details, OAuth URL, required scopes
```

### 4.3 Get User's Connected Integrations
```
GET /api/v1/integrations/connected
Response: List of user's connected integrations with status
```

### 4.4 Initiate OAuth Flow
```
GET /api/v1/integrations/{integration_id}/oauth/authorize
Response: { "oauth_url": "https://..." }
- Generates state parameter (CSRF protection)
- Stores state in session/cache
- Returns OAuth authorization URL
```

### 4.5 OAuth Callback Handler
```
GET /api/v1/integrations/{integration_id}/oauth/callback
Query Params: code, state, error, error_description
- Validates state parameter
- Exchanges code for tokens
- Stores tokens (encrypted) in database
- Updates user_integrations table
- Redirects to frontend success page
```

### 4.6 Disconnect Integration
```
POST /api/v1/integrations/{integration_id}/disconnect
- Revokes tokens (if revoke endpoint available)
- Updates status to "disconnected"
- Optionally deletes tokens
```

### 4.7 Refresh Token
```
POST /api/v1/integrations/{integration_id}/refresh
- Refreshes expired access token
- Updates tokens in database
- Returns new token info
```

### 4.8 Get Integration Status
```
GET /api/v1/integrations/{integration_id}/status
Response: Connection status, last used, token expiry, etc.
```

---

## 5. Security Requirements

### 5.1 Token Storage
- **Encrypt tokens at rest** (use AWS KMS or similar)
- Store refresh tokens separately from access tokens
- Never log tokens in plaintext

### 5.2 OAuth State Parameter
- Generate random state for each OAuth flow
- Store in Redis/session with expiration (5-10 minutes)
- Validate on callback to prevent CSRF

### 5.3 Token Refresh
- Automatically refresh tokens before expiry
- Background job to refresh expired tokens
- Handle refresh failures gracefully

### 5.4 Error Handling
- Handle OAuth errors (user denied, invalid code, etc.)
- Log errors without exposing sensitive data
- Provide user-friendly error messages

---

## 6. Implementation Components Needed

### 6.1 Database Models
- `UserIntegration` schema (Pydantic)
- `Integration` schema (Pydantic)
- DAO methods for CRUD operations

### 6.2 OAuth Service Layer
- Base OAuth handler (abstract class)
- Platform-specific OAuth handlers:
  - `GoogleOAuthHandler`
  - `MicrosoftOAuthHandler`
  - `SlackOAuthHandler`
  - `NotionOAuthHandler`
  - etc.

### 6.3 Token Management
- Token encryption/decryption service
- Token refresh service
- Token validation service

### 6.4 API Routes
- `integrations.py` route file
- All endpoints listed in section 4

### 6.5 Configuration
- Add integration configs to `settings.py`
- Environment variables for client IDs/secrets
- Integration metadata (can be JSON file or database)

---

## 7. Environment Variables Needed

```bash
# Google Workspace
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/google_workspace/callback

# Microsoft 365
MS_CLIENT_ID=...
MS_CLIENT_SECRET=...
MS_TENANT_ID=... (or "common")
MS_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/microsoft_365/callback

# Slack
SLACK_CLIENT_ID=...
SLACK_CLIENT_SECRET=...
SLACK_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/slack/callback

# Notion
NOTION_CLIENT_ID=...
NOTION_CLIENT_SECRET=...
NOTION_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/notion/callback

# Zoom
ZOOM_CLIENT_ID=...
ZOOM_CLIENT_SECRET=...
ZOOM_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/zoom/callback

# HubSpot
HUBSPOT_CLIENT_ID=...
HUBSPOT_CLIENT_SECRET=...
HUBSPOT_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/hubspot/callback

# Salesforce
SALESFORCE_CLIENT_ID=...
SALESFORCE_CLIENT_SECRET=...
SALESFORCE_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/salesforce/callback

# GitHub
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
GITHUB_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/github/callback

# Jira
JIRA_CLIENT_ID=...
JIRA_CLIENT_SECRET=...
JIRA_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/jira/callback

# Asana
ASANA_CLIENT_ID=...
ASANA_CLIENT_SECRET=...
ASANA_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/asana/callback

# Trello
TRELLO_API_KEY=...
TRELLO_API_SECRET=...
TRELLO_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/trello/callback

# Stripe
STRIPE_CLIENT_ID=...
STRIPE_CLIENT_SECRET=...
STRIPE_REDIRECT_URI=https://yourdomain.com/api/v1/integrations/stripe/callback

# Encryption Key (for token encryption)
INTEGRATION_TOKEN_ENCRYPTION_KEY=... (or use AWS KMS)
```

---

## 8. Implementation Priority

### Phase 1: Core Infrastructure
1. Database schema and models
2. Base OAuth handler
3. Token encryption service
4. Basic API endpoints

### Phase 2: High-Priority Integrations
1. Google Workspace (Calendar, Gmail)
2. Microsoft 365 (Outlook, Calendar)
3. Slack
4. Zoom

### Phase 3: Medium-Priority Integrations
5. Notion
6. HubSpot
7. Salesforce
8. GitHub

### Phase 4: Lower-Priority Integrations
9. Jira
10. Asana
11. Trello
12. Stripe
13. Others

---

## 9. Testing Requirements

- Unit tests for OAuth handlers
- Integration tests for OAuth flows
- Token encryption/decryption tests
- Error handling tests
- Security tests (CSRF, token leakage)

---

## 10. Documentation Needed

- API documentation for integration endpoints
- Setup guides for each integration
- OAuth flow diagrams
- Troubleshooting guide

