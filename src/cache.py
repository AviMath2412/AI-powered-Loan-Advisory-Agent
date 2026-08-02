import os
import json
import hashlib
import threading
from typing import Optional, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "query_response_cache")


class ResponseCache:
    """
    In-memory and disk-backed cache for query responses.
    Prevents redundant LLM graph executions for repeated queries.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def generate_key(
        self,
        prompt: str,
        model: str,
        provider: str,
        uploaded_doc_text: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Creates a unique deterministic SHA-256 key for a query + context payload.
        """
        normalized_prompt = prompt.strip().lower()
        doc_hash = hashlib.md5((uploaded_doc_text or "").encode("utf-8")).hexdigest() if uploaded_doc_text else "nodoc"
        profile_str = json.dumps(user_profile or {}, sort_keys=True)
        
        raw_key = f"{normalized_prompt}|{model}|{provider}|{doc_hash}|{profile_str}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _get_file_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a cached response dict if it exists.
        """
        with self._lock:
            if key in self._memory_cache:
                return self._memory_cache[key]

            file_path = self._get_file_path(key)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._memory_cache[key] = data
                        return data
                except Exception:
                    return None
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Saves a response dict to memory and disk cache.
        """
        with self._lock:
            self._memory_cache[key] = value
            file_path = self._get_file_path(key)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def clear(self) -> None:
        """
        Clears all in-memory and disk cache files.
        """
        with self._lock:
            self._memory_cache.clear()
            if os.path.exists(self.cache_dir):
                for fname in os.listdir(self.cache_dir):
                    if fname.endswith(".json"):
                        try:
                            os.remove(os.path.join(self.cache_dir, fname))
                        except Exception:
                            pass


# Global singleton instance
response_cache = ResponseCache()
