"""Per-user Gmail OAuth 2.0 client using the Google API directly.

Unlike gmail_tools.py (which spawns a single gmail-mcp subprocess for the
whole app), this module supports per-user authentication: each visitor
goes through Google's official consent screen and gets their own tokens.

Tokens are persisted to .hakathon/gmail_tokens/{session_id}.json so they
survive page refreshes and short-lived restarts.

Exposed functions:
    get_auth_url(session_id)  -> str          (Google consent screen URL)
    exchange_code(code, session_id) -> dict    (tokens dict, saves to disk)
    load_tokens(session_id) -> dict | None     (load from disk)
    delete_tokens(session_id) -> None          (disconnect)
    is_connected(session_id) -> bool
    get_gmail_service(session_id) -> Resource  (googleapiclient discovery)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

import config

log = logging.getLogger(__name__)

# Scopes required by this app (must match the consent screen config).
_SCOPES = config.GOOGLE_OAUTH_SCOPES


def _tokens_dir() -> Path:
    """Directory for per-user token files, created on first use."""
    d = config.GMAIL_TOKENS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _token_path(session_id: str) -> Path:
    return _tokens_dir() / f"{session_id}.json"


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------

def get_auth_url(session_id: str) -> str:
    """Build the Google OAuth consent-screen URL for `session_id`.

    The session_id is encoded as the OAuth `state` parameter so the callback
    can route tokens back to the correct Streamlit session.
    """
    if not config.GOOGLE_OAUTH_CLIENT_ID or not config.GOOGLE_OAUTH_CLIENT_SECRET:
        raise RuntimeError(
            "Google OAuth is not configured. Set GOOGLE_OAUTH_CLIENT_ID and "
            "GOOGLE_OAUTH_CLIENT_SECRET in your .env file."
        )

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
                "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [config.GOOGLE_OAUTH_REDIRECT_URI],
            }
        },
        scopes=_SCOPES,
    )
    flow.redirect_uri = config.GOOGLE_OAUTH_REDIRECT_URI

    url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        state=session_id,
        include_granted_scopes="true",
    )
    return url


def exchange_code(code: str, session_id: str) -> dict:
    """Exchange an authorization code for tokens; persist and return them."""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": config.GOOGLE_OAUTH_CLIENT_ID,
                "client_secret": config.GOOGLE_OAUTH_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [config.GOOGLE_OAUTH_REDIRECT_URI],
            }
        },
        scopes=_SCOPES,
    )
    flow.redirect_uri = config.GOOGLE_OAUTH_REDIRECT_URI
    flow.fetch_token(code=code)

    creds = flow.credentials
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    _save_tokens(session_id, token_data)
    log.info("Gmail OAuth tokens saved for session %s", session_id)
    return token_data


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------

def _save_tokens(session_id: str, token_data: dict) -> None:
    path = _token_path(session_id)
    path.write_text(json.dumps(token_data, indent=2), encoding="utf-8")


def load_tokens(session_id: str) -> Optional[dict]:
    path = _token_path(session_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def delete_tokens(session_id: str) -> None:
    path = _token_path(session_id)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def is_connected(session_id: str) -> bool:
    tokens = load_tokens(session_id)
    return tokens is not None and bool(tokens.get("token"))


# ---------------------------------------------------------------------------
# Gmail service (googleapiclient)
# ---------------------------------------------------------------------------

def get_gmail_service(session_id: str):
    """Return an authorized Gmail API service for the given session.

    Automatically refreshes expired tokens. Raises if not connected.
    """
    tokens = load_tokens(session_id)
    if not tokens:
        raise RuntimeError(f"No Gmail tokens for session {session_id}. Connect first.")

    creds = Credentials(
        token=tokens.get("token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri=tokens.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=tokens.get("client_id", config.GOOGLE_OAUTH_CLIENT_ID),
        client_secret=tokens.get("client_secret", config.GOOGLE_OAUTH_CLIENT_SECRET),
        scopes=tokens.get("scopes"),
    )

    # Auto-refresh if expired
    if creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
        # Persist refreshed token
        tokens["token"] = creds.token
        if creds.expiry:
            tokens["expiry"] = creds.expiry.isoformat()
        _save_tokens(session_id, tokens)

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Gmail API helpers (called by Streamlit tools)
# ---------------------------------------------------------------------------

def list_messages(session_id: str, query: str = "", max_results: int = 20) -> dict:
    """List messages in the user's inbox."""
    service = get_gmail_service(session_id)
    results = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    return results


def get_message(session_id: str, msg_id: str, format: str = "full") -> dict:
    """Get a single message by ID."""
    service = get_gmail_service(session_id)
    return service.users().messages().get(
        userId="me", id=msg_id, format=format
    ).execute()


def get_profile(session_id: str) -> dict:
    """Get the user's Gmail profile."""
    service = get_gmail_service(session_id)
    return service.users().getProfile(userId="me").execute()


def send_message(session_id: str, to: str, subject: str, body: str) -> dict:
    """Send an email message."""
    service = get_gmail_service(session_id)
    import base64
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    return service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()


def list_labels(session_id: str) -> dict:
    """List all labels in the user's mailbox."""
    service = get_gmail_service(session_id)
    return service.users().labels().list(userId="me").execute()


def list_threads(session_id: str, query: str = "", max_results: int = 20) -> dict:
    """List threads in the user's inbox."""
    service = get_gmail_service(session_id)
    return service.users().threads().list(
        userId="me", q=query, maxResults=max_results
    ).execute()


def get_thread(session_id: str, thread_id: str, format: str = "full") -> dict:
    """Get a single thread by ID."""
    service = get_gmail_service(session_id)
    return service.users().threads().get(
        userId="me", id=thread_id, format=format
    ).execute()


def create_draft(session_id: str, to: str, subject: str, body: str) -> dict:
    """Create a draft email."""
    service = get_gmail_service(session_id)
    import base64
    from email.mime.text import MIMEText

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    return service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}
    ).execute()


def trash_message(session_id: str, msg_id: str) -> dict:
    """Move a message to trash."""
    service = get_gmail_service(session_id)
    return service.users().messages().trash(
        userId="me", id=msg_id
    ).execute()


def delete_message(session_id: str, msg_id: str) -> dict:
    """Permanently delete a message."""
    service = get_gmail_service(session_id)
    return service.users().messages().delete(
        userId="me", id=msg_id
    ).execute()


# ---------------------------------------------------------------------------
# LangChain tools for per-user Gmail access
# ---------------------------------------------------------------------------

def create_user_gmail_tools(session_id: str) -> list:
    """Return LangChain tools that call the Gmail API with the user's tokens.

    These tools are created per-session and close over `session_id`, so the
    LLM can use Gmail tools that operate on the authenticated user's account.
    """
    from langchain_core.tools import tool

    @tool
    def gmail_get_profile() -> dict:
        """Get the authenticated Gmail user's profile (email address)."""
        return get_profile(session_id)

    @tool
    def gmail_list_messages(query: str = "", max_results: int = 20) -> dict:
        """List messages in the user's Gmail inbox. Use query syntax like 'is:unread', 'from:someone@example.com', etc."""
        return list_messages(session_id, query=query, max_results=max_results)

    @tool
    def gmail_get_message(msg_id: str) -> dict:
        """Get a single Gmail message by its ID. Returns full message content."""
        return get_message(session_id, msg_id)

    @tool
    def gmail_send_message(to: str, subject: str, body: str) -> dict:
        """Send an email via Gmail. Always confirm with the user before sending."""
        return send_message(session_id, to=to, subject=subject, body=body)

    @tool
    def gmail_list_threads(query: str = "", max_results: int = 20) -> dict:
        """List email threads in the user's inbox."""
        return list_threads(session_id, query=query, max_results=max_results)

    @tool
    def gmail_list_labels() -> dict:
        """List all labels/folders in the user's Gmail."""
        return list_labels(session_id)

    @tool
    def gmail_create_draft(to: str, subject: str, body: str) -> dict:
        """Create a draft email without sending it."""
        return create_draft(session_id, to=to, subject=subject, body=body)

    return [
        gmail_get_profile,
        gmail_list_messages,
        gmail_get_message,
        gmail_send_message,
        gmail_list_threads,
        gmail_list_labels,
        gmail_create_draft,
    ]
