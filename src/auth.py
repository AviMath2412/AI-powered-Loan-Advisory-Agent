import os
import sqlite3
import jwt
import bcrypt
import datetime
from typing import Optional, Dict, Any

AUTH_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "users.sqlite"
)
AUTH_DB_PATH = os.path.normpath(AUTH_DB_PATH)

# Use a secure secret key from environment or default for dev
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "super-secret-development-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def get_auth_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
    return sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)


def init_auth_db():
    conn = get_auth_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'USER'
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rate_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create a default admin user if no users exist
    cur = conn.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        _register_user_internal(conn, "admin", "admin123", "ADMIN")
        
    conn.commit()
    conn.close()


def _register_user_internal(conn: sqlite3.Connection, username: str, password: str, role: str = "USER") -> bool:
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hashed, role)
        )
        return True
    except sqlite3.IntegrityError:
        return False


def register_user(username: str, password: str, role: str = "USER") -> bool:
    conn = get_auth_connection()
    success = _register_user_internal(conn, username, password, role)
    conn.commit()
    conn.close()
    return success


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verifies user credentials. Returns a dict with user info if valid, else None.
    """
    conn = get_auth_connection()
    cur = conn.execute("SELECT id, username, password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return None
        
    user_id, db_username, password_hash, role = row
    if bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
        return {"user_id": user_id, "username": db_username, "role": role}
    return None


def generate_jwt(username: str, role: str) -> str:
    """Generates a JWT token for the authenticated user."""
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT token."""
    try:
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def check_rate_limit(username: str, limit: int = 20, window_minutes: int = 60) -> bool:
    """
    Checks if the user has exceeded their request limit within the given time window.
    Records the current request if they have not exceeded the limit.
    Returns True if allowed, False if rate limited.
    """
    conn = get_auth_connection()
    
    # Clean up old entries
    cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=window_minutes)
    conn.execute("DELETE FROM rate_limits WHERE timestamp < ?", (cutoff_time,))
    
    # Count recent requests
    cur = conn.execute("SELECT COUNT(*) FROM rate_limits WHERE username = ?", (username,))
    request_count = cur.fetchone()[0]
    
    if request_count >= limit:
        conn.commit()
        conn.close()
        return False
        
    # Log new request
    conn.execute("INSERT INTO rate_limits (username, timestamp) VALUES (?, ?)", (username, datetime.datetime.utcnow()))
    conn.commit()
    conn.close()
    
    return True

# Initialize the database on import
init_auth_db()
