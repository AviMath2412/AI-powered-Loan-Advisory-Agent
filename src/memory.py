import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Checkpoint URIs
CHECKPOINT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "checkpoints.sqlite"
)
CHECKPOINT_DB_PATH = os.path.normpath(CHECKPOINT_DB_PATH)

POSTGRES_URI = os.getenv("POSTGRES_URI", "")

_checkpointer = None
_pg_pool = None

def get_checkpointer():
    """
    Returns a process-wide singleton Saver.
    Uses PostgreSQL if POSTGRES_URI is provided, otherwise falls back to SQLite.
    Requires `langgraph-checkpoint-postgres` and `psycopg_pool` for Postgres.
    """
    global _checkpointer, _pg_pool
    
    if _checkpointer is not None:
        return _checkpointer

    if POSTGRES_URI:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool
            
            _pg_pool = ConnectionPool(conninfo=POSTGRES_URI, max_size=20)
            _checkpointer = PostgresSaver(_pg_pool)
            _checkpointer.setup() # Ensure tables exist
        except ImportError as e:
            raise ImportError(
                "PostgreSQL checkpointer requires 'langgraph-checkpoint-postgres' and 'psycopg_pool'. "
                "Install with: pip install langgraph-checkpoint-postgres psycopg psycopg-pool"
            ) from e
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
        
    return _checkpointer


def list_thread_ids(username: str = None) -> list[str]:
    """Used by the UI's session picker to list past conversations. Scoped by user if multi-tenancy is active."""
    if POSTGRES_URI:
        import psycopg
        try:
            with psycopg.connect(POSTGRES_URI) as conn:
                with conn.cursor() as cur:
                    if username:
                        cur.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE %s ORDER BY thread_id DESC", (f"{username}_%",))
                    else:
                        cur.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
                    return [row[0] for row in cur.fetchall()]
        except Exception:
            return []
    else:
        if not os.path.exists(CHECKPOINT_DB_PATH):
            return []
        try:
            conn = sqlite3.connect(CHECKPOINT_DB_PATH)
            if username:
                cur = conn.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ? ORDER BY thread_id DESC", (f"{username}_%",))
            else:
                cur = conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
            threads = [row[0] for row in cur.fetchall()]
            conn.close()
            return threads
        except sqlite3.Error:
            return []


def delete_thread(thread_id: str, username: str = None) -> bool:
    """Deletes all checkpoints and data for a given thread_id from the database. Enforces multi-tenancy if username provided."""
    if username and not thread_id.startswith(f"{username}_"):
        return False # Unauthorized
        
    if POSTGRES_URI:
        import psycopg
        try:
            with psycopg.connect(POSTGRES_URI) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                conn.commit()
            return True
        except Exception:
            return False
    else:
        if not os.path.exists(CHECKPOINT_DB_PATH):
            return False
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
            return True
        except sqlite3.Error:
            return False
