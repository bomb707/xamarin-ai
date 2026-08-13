"""Optional-dependency guard for the real (live) adapters.

Core replay/mock code has zero network dependencies. Real adapters need
`pip install xamarinbot[live]` (httpx + websockets); importing them without
that extra fails loudly here rather than at some deeper, confusing point.
"""
from __future__ import annotations

try:
    import httpx  # noqa: F401
    import websockets  # noqa: F401

    LIVE_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    LIVE_DEPS_AVAILABLE = False


def require_live_deps() -> None:
    if not LIVE_DEPS_AVAILABLE:
        raise ImportError(
            "Live adapters require the optional 'live' extra: "
            "pip install -e '.[live]' (httpx + websockets)."
        )
