"""
health_pipeline.security.auth
Token-based authentication, API key validation, and rate limiting for secure prediction endpoints.
"""

import hmac
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from health_pipeline.config import SECURITY_CONFIG


class Authenticator:
    """
    Handles API authorization and rate-limiting to prevent unauthorized access
    and denial-of-service attempts on medical prediction endpoints.
    """

    def __init__(
        self, 
        valid_tokens: Optional[List[str]] = None,
        rate_limit_per_minute: int = 60
    ):
        tokens = valid_tokens or [SECURITY_CONFIG.API_AUTH_TOKEN]
        self._valid_tokens = set(t.strip() for t in tokens if t)
        self.rate_limit_per_minute = rate_limit_per_minute
        self._request_history: Dict[str, List[float]] = defaultdict(list)

    def validate_token(self, token_header: Optional[str]) -> Tuple[bool, str]:
        """
        Validates an Authorization header (Bearer <token>) or raw API key
        using constant-time HMAC comparison to prevent timing side-channel attacks.
        """
        if not token_header:
            return False, "Missing authentication credentials."

        # Support 'Bearer <token>' or raw token string
        provided_token = token_header
        if token_header.startswith("Bearer "):
            provided_token = token_header[7:].strip()
        else:
            provided_token = token_header.strip()

        # Constant-time comparison across allowed keys
        for valid_token in self._valid_tokens:
            if hmac.compare_digest(provided_token.encode("utf-8"), valid_token.encode("utf-8")):
                return True, "Authenticated"

        return False, "Invalid or expired authentication token."

    def check_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Sliding-window rate limiter. Returns (is_allowed, remaining_requests).
        """
        now = time.time()
        one_minute_ago = now - 60.0

        # Purge requests older than 1 minute
        history = [ts for ts in self._request_history[client_id] if ts > one_minute_ago]
        self._request_history[client_id] = history

        if len(history) >= self.rate_limit_per_minute:
            return False, 0

        history.append(now)
        remaining = self.rate_limit_per_minute - len(history)
        return True, remaining
