import os
import time
import logging
import functools
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def setup(log_dir):
    """
    Call once per process to enable profiling output.
    Writes to log_dir/profile_{pid}.log.
    If never called all profiling functions are silent no-ops.
    """
    from pathlib import Path
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(log_dir / f'profile_{os.getpid()}.log')
    h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def _emit(msg):
    if logger.handlers:
        logger.debug(msg)


def timed(func):
    """Decorator: log pid, qualified function name, t0, and elapsed per call."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _emit(f"{os.getpid()} {func.__qualname__} {t0:.6f} {elapsed:.6f}")
        return result
    return wrapper


@contextmanager
def cl_timed(op, **kwargs):
    """
    Context manager for timing OpenCL operations by wall clock.
    Call queue.finish() inside the block to ensure GPU work is complete
    before the timer stops.

    Example:
        with cl_timed('h2d', nbytes=A.nbytes):
            cl.enqueue_copy(queue, buf, A)
            queue.finish()
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    extra = ' '.join(f'{k}={v}' for k, v in kwargs.items())
    _emit(f"{os.getpid()} cl:{op} {t0:.6f} {elapsed:.6f} {extra}")
