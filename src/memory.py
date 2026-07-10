import sqlite3
import os
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver

# Define absolute path to database in the project data/ directory
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/checkpoints.sqlite"))

def get_checkpointer() -> SqliteSaver:
    """
    Initializes and returns the SqliteSaver checkpointer for LangGraph.
    Ensures that the target database file and schema are set up.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer

def list_thread_ids() -> list[str]:
    """
    Queries the database and returns a list of all unique thread IDs.
    """
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Verify if table exists before querying
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';")
        if not cursor.fetchone():
            conn.close()
            return []
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
        thread_ids = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return thread_ids
    except Exception:
        return []
