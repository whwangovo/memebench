"""Utility helpers."""

__all__ = ["retry_api_call", "retry_sync"]


def __getattr__(name):
    if name in {"retry_api_call", "retry_sync"}:
        from .retry import retry_api_call, retry_sync

        return {"retry_api_call": retry_api_call, "retry_sync": retry_sync}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
