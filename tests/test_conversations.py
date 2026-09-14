import tempfile
from pathlib import Path
import pytest
from app.storage.conversations import ConversationStore


def test_conversation_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_conversations.db"
        store = ConversationStore(db_path=db_file)

        # 1. Create conversation
        cid = store.create_conversation(title="Test Session")
        assert cid is not None

        # 2. Add user and assistant messages
        store.add_message(cid, role="user", content="What is agentic RAG?")
        store.add_message(
            cid,
            role="assistant",
            content="Agentic RAG uses reasoning loops [S1].",
            sources=[{"source_id": "S1", "url": "https://example.com"}],
        )

        # 3. Retrieve conversation
        conv = store.get_conversation(cid)
        assert conv is not None
        assert conv["title"] == "Test Session"
        assert len(conv["messages"]) == 2
        assert conv["messages"][0]["role"] == "user"
        assert conv["messages"][1]["role"] == "assistant"
        assert conv["messages"][1]["sources"][0]["source_id"] == "S1"

        # 4. Check recent history formatting
        history = store.get_recent_chat_history(cid, max_turns=2)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

        # 5. List conversations
        conv_list = store.list_conversations()
        assert len(conv_list) == 1
        assert conv_list[0]["id"] == cid

        # 6. Delete conversation
        deleted = store.delete_conversation(cid)
        assert deleted is True
        assert store.get_conversation(cid) is None
        assert len(store.list_conversations()) == 0
