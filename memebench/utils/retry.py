"""Async retry utility with exponential backoff for API calls."""

import asyncio
import logging

logger = logging.getLogger(__name__)


def _is_bad_request_error(error: Exception) -> bool:
    return error.__class__.__name__ == "BadRequestError"


async def retry_api_call(coro_factory, max_retries=3, base_delay=2.0):
    """Retry an async API call with exponential backoff.

    Args:
        coro_factory: Zero-argument callable that returns a new coroutine each time.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).

    Returns:
        The result of the coroutine, or None if all retries fail.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            if _is_bad_request_error(e):
                # 400 errors (e.g. content filter) are not retryable
                print(f"  [skip] BadRequestError (not retrying): {e}")
                return None
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                print(
                    f"  [retry] attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    print(f"  [retry] All {max_retries + 1} attempts failed. Last error: {last_error}")
    return None


def retry_sync(func, max_retries=3, base_delay=2.0):
    """Retry a synchronous call with exponential backoff.

    Args:
        func: Zero-argument callable to retry.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (doubles each retry).

    Returns:
        The result of the function, or None if all retries fail.
    """
    import time

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(
                    f"  [retry] attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    print(f"  [retry] All {max_retries + 1} attempts failed. Last error: {last_error}")
    return None
