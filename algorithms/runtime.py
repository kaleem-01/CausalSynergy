# runtime.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar
import time
import sys
import gc

T = TypeVar("T")

# Optional: psutil gives the most reliable RSS measurement cross-platform.
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# Unix-only: resource can provide ru_maxrss (process peak RSS since start).
try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


@dataclass
class RuntimeStats:
    wall_time_s: float
    cpu_time_s: float

    # RSS (resident set size) snapshots (best effort)
    rss_before_bytes: Optional[int] = None
    rss_after_bytes: Optional[int] = None
    rss_delta_bytes: Optional[int] = None

    # Peak RSS since process start (Unix best effort; monotonically nondecreasing)
    ru_maxrss_bytes: Optional[int] = None

    # Peak Python allocation (only if tracemalloc=True)
    tracemalloc_peak_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_rss_bytes() -> Optional[int]:
    """
    Current process RSS in bytes (best effort).
    - Prefer psutil if available.
    - Otherwise return None (we avoid parsing /proc to keep it simple/portable).
    """
    if psutil is None:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _get_ru_maxrss_bytes() -> Optional[int]:
    """
    Peak RSS since process start (Unix only), in bytes (best effort).
    Note: ru_maxrss unit differs by OS:
      - Linux: kilobytes
      - macOS: bytes
    """
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        ru_maxrss = int(getattr(usage, "ru_maxrss", 0))
        if ru_maxrss <= 0:
            return None

        # Convert to bytes
        if sys.platform == "darwin":
            return ru_maxrss  # already bytes
        return ru_maxrss * 1024  # Linux: KB -> bytes
    except Exception:
        return None


def measure_call(
    fn: Callable[..., T],
    *args: Any,
    tracemalloc: bool = False,
    gc_collect: bool = False,
    **kwargs: Any,
) -> Tuple[T, RuntimeStats]:
    """
    Measure wall-time, CPU-time, RSS before/after (if possible), and optional peaks.

    Parameters
    ----------
    tracemalloc : bool
        If True, captures peak Python allocation during fn call (not full RSS).
    gc_collect : bool
        If True, runs gc.collect() before measuring (reduces noise, slightly slower).
    """
    if gc_collect:
        gc.collect()

    rss_before = _get_rss_bytes()
    peak_before = _get_ru_maxrss_bytes()

    t0 = time.perf_counter()
    c0 = time.process_time()

    if tracemalloc:
        import tracemalloc as _tracemalloc  # local import to avoid overhead
        _tracemalloc.start()

    try:
        result = fn(*args, **kwargs)
    except Exception:
        # Still stop tracemalloc if enabled, then re-raise.
        if tracemalloc:
            import tracemalloc as _tracemalloc
            _tracemalloc.stop()
        raise

    if tracemalloc:
        import tracemalloc as _tracemalloc
        _current, peak = _tracemalloc.get_traced_memory()
        _tracemalloc.stop()
        tm_peak = int(peak)
    else:
        tm_peak = None

    t1 = time.perf_counter()
    c1 = time.process_time()

    rss_after = _get_rss_bytes()
    peak_after = _get_ru_maxrss_bytes()

    if rss_before is not None and rss_after is not None:
        rss_delta = int(rss_after - rss_before)
    else:
        rss_delta = None

    # ru_maxrss is monotone within a process; report the post-call value
    # (and keep it as "best effort peak RSS since start").
    stats = RuntimeStats(
        wall_time_s=float(t1 - t0),
        cpu_time_s=float(c1 - c0),
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_after,
        rss_delta_bytes=rss_delta,
        ru_maxrss_bytes=peak_after if peak_after is not None else peak_before,
        tracemalloc_peak_bytes=tm_peak,
    )
    return result, stats


def bytes_to_mb(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    return float(x) / (1024.0 * 1024.0)
