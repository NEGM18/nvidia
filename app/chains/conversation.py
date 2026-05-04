"""
Conversation Manager — Tracks chat history per session.

Uses LangChain's ChatMessageHistory for in-memory session tracking
with RunnableWithMessageHistory for transparent history injection.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# In-memory store for session histories
_session_histories: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Get or create a chat history for the given session.

    Args:
        session_id: Unique identifier for the chat session.

    Returns:
        ChatMessageHistory instance for the session.
    """
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def wrap_with_history(chain):
    """
    Wrap a chain with message history tracking.

    Args:
        chain: A LangChain runnable chain.

    Returns:
        A chain that automatically manages conversation history.
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def clear_session(session_id: str) -> None:
    """Clear the chat history for a specific session."""
    if session_id in _session_histories:
        _session_histories[session_id].clear()


def clear_all_sessions() -> None:
    """Clear all chat histories."""
    _session_histories.clear()


def get_session_messages(session_id: str) -> list[dict]:
    """
    Get all messages for a session as a list of dicts.

    Returns:
        List of {"role": "human"|"ai", "content": "..."} dicts.
    """
    if session_id not in _session_histories:
        return []

    messages = _session_histories[session_id].messages
    return [
        {
            "role": "human" if msg.type == "human" else "ai",
            "content": msg.content,
        }
        for msg in messages
    ]


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_session_histories.keys())
