"""Task-local prohibition of paid provider clients in subscription workflows."""
from contextlib import contextmanager
from contextvars import ContextVar


_SUBSCRIPTION_ONLY = ContextVar("subscription_only", default=False)


def require_api_allowed():
    if _SUBSCRIPTION_ONLY.get():
        raise RuntimeError("Paid API calls are forbidden in subscription mode")


@contextmanager
def subscription_only(enabled=True):
    token = _SUBSCRIPTION_ONLY.set(enabled or _SUBSCRIPTION_ONLY.get())
    try:
        yield
    finally:
        _SUBSCRIPTION_ONLY.reset(token)
