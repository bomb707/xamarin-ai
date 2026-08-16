#!/usr/bin/env python3
"""LIVE real-market SHADOW service - Strategy V0, PAPER ONLY.

This is the production entrypoint the readiness audit's item 4C asked for:
neither "real recorder" (which never calls the strategy) nor "replay shadow"
(which needs a finished capture), but the live bot.

    LIVE Polymarket CLOB + Chainlink/TWAP/Binance
        -> incremental normalized projection
        -> FeatureEngine
        -> Strategy V0 (regime, candidates, OneStepController)
        -> TradingSession  (PAPER orders only)
        -> real subsequent market evolution
        -> settlement
        -> permanent shadow journal

SAFETY. No private key is read, no authenticated CLOB client is imported,
nothing is signed, and no order, cancel or replacement is ever sent. The
only "dispatch" mutates `TradingSession`'s in-process paper state. This is
enforced structurally by `tests/test_import_boundaries.py`, which walks this
file's entire transitive import graph.

    # three rounds, then stop
    python scripts/run_real_shadow.py --rounds 3

    # keep going
    python scripts/run_real_shadow.py --rounds 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.realtime.identity import RecorderIdentity  # noqa: E402
from xamarinbot.realtime.lifecycle import LifecycleConfig  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402
from xamarinbot.realtime.service import RealRecorderService, ServiceConfig  # noqa: E402
from xamarinbot.shadow.journal import ShadowJournal  # noqa: E402
from xamarinbot.shadow.live import LiveShadowService  # noqa: E402
from xamarinbot.shadow.manifest import NO_REAL_MODEL  # noqa: E402

SHADOW_DIR = _REPO_ROOT / "captures" / "shadow"


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sweep", type=float, default=900.0,
                    help="seconds to poll for venue resolutions after the batch")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = utc()
    capture_db = SHADOW_DIR / f"shadow_raw_{stamp}.db"
    journal_db = SHADOW_DIR / f"shadow_journal_{stamp}.db"

    identity = RecorderIdentity.capture(_REPO_ROOT)
    log = (lambda *a: None) if args.quiet else print

    print("=" * 78)
    print("LIVE REAL-MARKET SHADOW - STRATEGY V0 - PAPER ONLY")
    print("  No private key. No authenticated client. No order is ever sent.")
    print(f"  code sha     {identity.recorder_code_sha}"
          f"{'  (DIRTY)' if identity.recorder_code_dirty else ''}")
    print(f"  pid          {identity.process_pid}   started {utc(identity.process_started_at)}")
    print(f"  rounds       {args.rounds}")
    print(f"  raw capture  {capture_db.relative_to(_REPO_ROOT)}")
    print(f"  journal      {journal_db.relative_to(_REPO_ROOT)}")
    print(f"  model        {NO_REAL_MODEL}")
    print("=" * 78, flush=True)

    store = RawEventStore(str(capture_db))
    journal = ShadowJournal(str(journal_db))
    service = RealRecorderService(
        store,
        ServiceConfig(n_rounds=args.rounds, lifecycle=LifecycleConfig(),
                      resolution_sweep_s=args.sweep),
        log=log, identity=identity,
    )
    # No model is passed: there is no Gate-A-frozen REAL model, so every
    # ALPHA decision is correctly blocked as MODEL_UNAVAILABLE. This is
    # MODE A (pre-model live shadow) - an operational pipeline test, NOT a
    # profitability measurement.
    shadow = LiveShadowService(service, journal, model=None, feature_set=None, log=print)

    print(f"[shadow] strategy {shadow.manifest.strategy_version} "
          f"config_hash={shadow.manifest.config_hash} "
          f"grid={len(shadow.grid)} points ({shadow.grid[0]:.0f}s..{shadow.grid[-1]:.0f}s)",
          flush=True)

    try:
        service.run()
    finally:
        store.close()

    summary = shadow.summary()
    summary["counts"] = journal.counts()
    summary["journal_db"] = str(journal_db.relative_to(_REPO_ROOT))
    summary["raw_capture_db"] = str(capture_db.relative_to(_REPO_ROOT))
    (SHADOW_DIR / f"shadow_summary_{stamp}.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    journal.close()

    print("\n" + "=" * 78)
    print("SHADOW SUMMARY")
    print(json.dumps(summary, indent=2))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
