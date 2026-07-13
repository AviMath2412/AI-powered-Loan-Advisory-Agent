import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

# Lives alongside your chroma_db, in data/
CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "checkpoints.sqlite"
)
CHECKPOINT_DB_PATH = os.path.normpath(CHECKPOINT_DB_PATH)

_checkpointer = None


def get_checkpointer() -> SqliteSaver:
    """
    Returns a process-wide singleton SqliteSaver so Streamlit reruns (which re-execute the
    whole script) don't reopen a new connection every turn. Requires `langgraph-checkpoint-sqlite`.
    """
    global _checkpointer
    if _checkpointer is None:
        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    return _checkpointer


def list_thread_ids() -> list[str]:
    """Used by the UI's session picker to list past conversations."""
    if not os.path.exists(CHECKPOINT_DB_PATH):
        return []
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        cur = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
        )
        threads = [row[0] for row in cur.fetchall()]
        conn.close()
        return threads
    except sqlite3.Error:
        return []


def delete_thread(thread_id: str):
    """Deletes all checkpoints and data for a given thread_id from the database."""
    if not os.path.exists(CHECKPOINT_DB_PATH):
        return
    try:
        conn = sqlite3.connect(CHECKPOINT_DB_PATH)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        try:
            conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
        except sqlite3.Error:
            pass
        try:
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
        except sqlite3.Error:
            pass
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass
