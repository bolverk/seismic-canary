"""Static calibration check: does the model separate known explosions from
known earthquakes?

This is deliberately different from scripts/validate_model.py, which pulls
a large sample dynamically from the live USGS catalog. Here the calibration
set is small, fixed, and individually documented -- the same events shown in
the "Validation Against Known Explosions" section of pages/methodology.py.

Both this script and that page import the calibration set and run it
through the model via src/models/calibration.py, so there is exactly one
source of truth: the page can never drift out of sync with what the model
actually does.

Usage:
    python scripts/validate_calibration_set.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.calibration import run_calibration, summarize
from src.models.anomaly import RuleBasedModel


def main() -> bool:
    model = RuleBasedModel()
    results = run_calibration(model)

    print(f"Model version: {model.get_version()}\n")

    header = f"{'Event':<28} {'Category':<22} {'Exp':>3} {'Got':>3} {'Ex.score':>8} {'Eq.score':>8}  Verified  Result"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.event.name:<28} {r.event.true_category:<22} {r.event.expected_alert_level:>3} "
            f"{r.alert_level:>3} {r.explosion_consistency:>8.3f} "
            f"{r.earthquake_consistency:>8.3f}  "
            f"{'yes' if r.event.verified else 'est.':<8}  {'PASS' if r.passed else 'FAIL'}"
        )

    print()
    summary = summarize(results)
    for category, (total, passed) in summary.items():
        if category == "overall":
            continue
        print(f"  {category:<22} {passed}/{total} correctly classified")

    # Demonstrate why ground-truth calibration matters, for any event that
    # carries a naive/raw-catalog counterpart input.
    for r in results:
        if r.naive_alert_level is None:
            continue
        print(f"\nNaive vs. calibrated input, {r.event.name}:")
        print(f"  naive (raw catalog values) -> level={r.naive_alert_level}  "
              f"explosion_consistency={r.naive_explosion_consistency:.3f}")
        print(f"  calibrated (ground truth)  -> level={r.alert_level}  "
              f"explosion_consistency={r.explosion_consistency:.3f}")

    total, passed = summary["overall"]
    return passed == total


if __name__ == "__main__":
    ok = main()
    print(f"\n{'ALL CALIBRATION EVENTS CORRECTLY CLASSIFIED' if ok else 'CALIBRATION FAILURES DETECTED'}")
    sys.exit(0 if ok else 1)
