from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from ._log import log

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exc_cls: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> T:
    """Run a callable, retrying transient failures with exponential backoff.

    The callable is attempted up to ``max_attempts`` times. If it raises one of
    the exception types specified by ``exc_cls``, the failure is logged and the
    function waits before trying again. The wait duration grows exponentially
    from ``base_delay`` and is capped at ``max_delay``.

    This function uses "full jitter" for retry delays: instead of sleeping for
    the exact exponential delay, it sleeps for a random duration between zero and
    the capped delay. This helps avoid many callers retrying at the same time.

    Args:
        fn: Zero-argument callable to run.
        max_attempts: Maximum number of total attempts, including the first
            call. Must be at least 1.
        base_delay: Initial delay, in seconds, used as the base for exponential
            backoff. Must be non-negative.
        max_delay: Maximum delay, in seconds, for any single sleep. Must be
            non-negative.
        exc_cls: Exception class, or tuple of exception classes, that should
            trigger a retry.

    Returns:
        The value returned by ``fn``.

    Raises:
        ValueError: If ``max_attempts`` is less than 1, or if either delay is
            negative.
        BaseException: Re-raises the final caught exception if all retry attempts
            fail. Exceptions not matching ``exc_cls`` are raised immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except exc_cls:
            # Do not sleep after the final failed attempt, preserve and re-raise
            # the original exception with its traceback.
            if attempt == max_attempts:
                raise

            # Grow the retry window exponentially: base, 2x base, 4x base, etc.
            exponential_delay = base_delay * 2 ** (attempt - 1)

            # Keep pathological retry schedules from sleeping for too long.
            capped_delay = min(exponential_delay, max_delay)

            # Full jitter spreads retry attempts across the whole capped window,
            # reducing the chance that many callers retry simultaneously.
            sleep = random.uniform(0, capped_delay)

            log.info(
                "Action failed. Retrying in %.2fs. Attempt %s/%s.",
                sleep,
                attempt,
                max_attempts,
                exc_info=True,
            )
            time.sleep(sleep)

    # The loop either returns from fn() or raises from the except block.
    raise RuntimeError("unreachable")
