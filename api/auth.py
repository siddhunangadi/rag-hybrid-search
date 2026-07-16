"""API key authentication, role-based authorization, and per-identity rate limiting.

Centralized here so every route imports the same ``require_role`` dependency
rather than re-implementing key checks. See ``Settings.api_keys`` docstring
for the no-op-when-unset dev fallback and the production requirement.
"""

import hashlib
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import Depends, Header, HTTPException, Request

from api.dependencies import Container, get_container
from rag_hybrid_search.audit import AuditEvent, now_utc

Role = Literal["admin", "reader"]

_ANONYMOUS_KEY_ID = "anonymous"


def _record_auth_failure(
    container: Container, request: Request, request_id: str, key_id: str, action: str
) -> None:
    """Record an auth-failure audit event, isolated so logging never masks the 401/403."""
    try:
        container.audit_log.record(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type="auth_failure",
                timestamp=now_utc(),
                request_id=request_id,
                key_id=key_id,
                endpoint=request.url.path,
                action=action,
                status="failure",
            )
        )
    except Exception:  # noqa: BLE001 - never let audit logging break the 401/403
        pass


@dataclass(frozen=True)
class Identity:
    key_id: str
    role: Role
    request_id: str


def get_identity(
    request: Request,
    container: Container = Depends(get_container),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Identity:
    """Resolve the caller's identity, enforcing auth and rate limits.

    No keys configured (``Settings.api_keys`` empty) -> every request is
    treated as admin, rate-limited per client IP, matching this project's
    existing "unset config = open dev default" convention.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    api_keys = container.settings.api_keys_by_key
    if not api_keys:
        identifier = request.client.host if request.client else _ANONYMOUS_KEY_ID
        container.rate_limiter.check(identifier)
        identity = Identity(key_id=_ANONYMOUS_KEY_ID, role="admin", request_id=request_id)
        request.state.identity = identity
        return identity

    if not x_api_key:
        _record_auth_failure(container, request, request_id, _ANONYMOUS_KEY_ID, "missing_api_key")
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    role = api_keys.get(x_api_key)
    if role is None:
        key_id = hashlib.sha256(x_api_key.encode()).hexdigest()[:12]
        _record_auth_failure(container, request, request_id, key_id, "invalid_api_key")
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

    key_id = hashlib.sha256(x_api_key.encode()).hexdigest()[:12]
    container.rate_limiter.check(key_id)
    identity = Identity(key_id=key_id, role=role, request_id=request_id)
    request.state.identity = identity
    return identity


def require_role(*allowed: Role):
    """FastAPI dependency factory restricting a route to the given roles."""

    def dependency(
        request: Request,
        container: Container = Depends(get_container),
        identity: Identity = Depends(get_identity),
    ) -> Identity:
        if identity.role not in allowed:
            _record_auth_failure(
                container, request, identity.request_id, identity.key_id, "role_denied"
            )
            raise HTTPException(status_code=403, detail=f"role {identity.role!r} not permitted")
        return identity

    return dependency
