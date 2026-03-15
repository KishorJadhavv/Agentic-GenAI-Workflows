import redis
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SessionState(BaseModel):
    """Represents the state of a workflow session."""
    session_id: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RedisSessionManager:
    """
    Enterprise Redis-backed session manager for multi-agent workflows.
    Ensures context persistence across distributed or long-running executions.
    """

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis_client.ping()
        except redis.ConnectionError:
            logger.warning("Redis not available. Falling back to in-memory session management.")
            self.redis_client = None
            self._local_storage: Dict[str, str] = {}

    def save_session(self, state: SessionState, ttl: int = 3600):
        """Persists the session state."""
        data = state.json()
        if self.redis_client:
            self.redis_client.setex(f"session:{state.session_id}", ttl, data)
        else:
            self._local_storage[state.session_id] = data

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieves the session state."""
        if self.redis_client:
            data = self.redis_client.get(f"session:{session_id}")
        else:
            data = self._local_storage.get(session_id)

        if not data:
            return None
        
        return SessionState.parse_raw(data)

    def clear_session(self, session_id: str):
        """Removes the session data."""
        if self.redis_client:
            self.redis_client.delete(f"session:{session_id}")
        else:
            self._local_storage.pop(session_id, None)

class InMemorySessionManager:
    """Lightweight in-memory alternative for local development."""
    
    def __init__(self):
        self._storage: Dict[str, SessionState] = {}

    def save_session(self, state: SessionState):
        self._storage[state.session_id] = state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._storage.get(session_id)
