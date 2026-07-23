"""The pacer is carved out of the council-infra PR as experimental: disabled by
default, and it must warn loudly if anyone enables it. This guards the marker
so it can't be silently dropped (see docs/pacer-rework.md)."""
from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proofstack.subscription import SubscriptionPacer, SubscriptionStore  # noqa: E402


class ExperimentalGateTests(unittest.TestCase):
    def _pacer(self, td: str) -> SubscriptionPacer:
        return SubscriptionPacer(SubscriptionStore(home=Path(td)), provider="claude")

    def test_disabled_by_default_and_silent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pacer = self._pacer(td)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                enabled, _ = pacer.gate_config()
            self.assertFalse(enabled)
            self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])

    def test_enabling_warns_experimental(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = SubscriptionStore(home=Path(td))
            store.save_settings({"enabled": True})
            pacer = SubscriptionPacer(store, provider="claude")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                enabled, _ = pacer.gate_config()
            self.assertTrue(enabled)
            msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
            self.assertTrue(any("EXPERIMENTAL" in m for m in msgs), msgs)


if __name__ == "__main__":
    unittest.main()
