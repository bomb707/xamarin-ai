#!/usr/bin/env python3
"""REAL_REPLAY smoke test (Phase 12C.1 item 18).

Projects a real Phase 12C capture through

    RawEventStore_real -> NormalizedEventStore_real -> FeatureEngine

and prints the acceptance evidence: projected event counts, the market
constraints as READ FROM THE MARKET, the FeatureVector count with the
histogram of invalid reasons, and a provenance proof that not one projected
event is synthetic.

Read-only with respect to the capture. Places no orders.

    python scripts/run_real_replay_smoke.py captures/phase12c_verify.db
"""
from __future__ import annotations

import argparse
import collections
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from xamarinbot.events.store import EventStore  # noqa: E402
from xamarinbot.events.types import EventType  # noqa: E402
from xamarinbot.features.config import FeatureConfig  # noqa: E402
from xamarinbot.features.engine import compute  # noqa: E402
from xamarinbot.features.types import FeatureVector  # noqa: E402
from xamarinbot.market.constraints import MarketConstraints  # noqa: E402
from xamarinbot.provenance import DataProvenance, describe  # noqa: E402
from xamarinbot.realtime.raw_store import RawEventStore  # noqa: E402
from xamarinbot.replay.feeds import market_config_from_payload  # noqa: E402
from xamarinbot.replay.projection import PROVENANCE_KEY, project_round  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", nargs="?", default="captures/phase12c_verify.db")
    ap.add_argument("--round", default=None, help="round id (default: the first in the capture)")
    ap.add_argument("--step", type=float, default=10.0, help="decision-point spacing, seconds")
    ap.add_argument("--out", default=None, help="where to write the projected store")
    args = ap.parse_args()

    capture = Path(args.capture)
    if not capture.exists():
        print(
            f"[smoke] capture {capture} not found.\n"
            "        Capture databases are gitignored (a 3-round capture is ~500MB).\n"
            "        Record one with: python scripts/run_real_recorder.py --rounds 1",
            file=sys.stderr,
        )
        return 77  # conventional "skipped"

    raw = RawEventStore(str(capture))
    round_id = args.round or (raw.round_ids() or [None])[0]
    if round_id is None:
        print(f"[smoke] {capture} contains no rounds", file=sys.stderr)
        return 1

    out_path = args.out or str(Path(tempfile.gettempdir()) / f"real_replay_{round_id}.db")
    Path(out_path).unlink(missing_ok=True)
    out = EventStore(out_path, provenance=DataProvenance.REAL_REPLAY)

    print("=" * 78)
    print("PHASE 12C.1 - REAL_REPLAY SMOKE TEST")
    print(f"provenance: {describe(out.provenance)}")
    print("=" * 78)
    print(f"capture      {capture}")
    print(f"round        {round_id}")
    print(f"projected to {out_path}")

    result = project_round(raw, round_id, out)

    print("\nPROJECTED EVENT COUNTS")
    print("-" * 78)
    for name, n in sorted(result.counts.items()):
        print(f"  {name:<18} {n:>8}")
    print(f"  {'TOTAL':<18} {result.total_projected:>8}")
    print(f"  settlement basis   {result.settlement_topic} (from the market's own metadata)")
    print(f"  p0                 {result.p0!r}  (real observation at/before the open)")

    print("\nRAW EVENTS DELIBERATELY NOT PROJECTED")
    print("-" * 78)
    if result.skipped:
        for reason, n in sorted(result.skipped.items()):
            print(f"  {reason:<44} {n:>8}")
    else:
        print("  (none)")
    for w in result.warnings:
        print(f"  ! {w}")

    # ------------------------------------------------ market constraints
    events = out.all_events(round_id)
    config_payload = next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)
    constraints = MarketConstraints.from_market_config(
        market_config_from_payload(config_payload),
        provenance=DataProvenance.REAL_REPLAY,
        source="projected MARKET_CONFIG",
    )
    print("\nEXECUTABLE MARKET CONSTRAINTS (read from the market, not guessed)")
    print("-" * 78)
    print(f"  min_order_shares   {constraints.min_order_shares}   (SHARES, not USDC notional)")
    print(f"  tick_size          {constraints.tick_size}")
    print(f"  fee rate           {constraints.fee_configuration.crypto_fee_rate}")
    print(f"  taker delay        {constraints.taker_delay_ms} ms")
    print(f"  settlement         {constraints.settlement_kind} / {constraints.twap_window_s}s")

    # ------------------------------------------------------- provenance
    print("\nPROVENANCE PROOF")
    print("-" * 78)
    missing = synthetic = 0
    for e in events:
        block = e.payload.get(PROVENANCE_KEY)
        if not block:
            missing += 1
        elif block.get("provenance") != DataProvenance.REAL_REPLAY.value:
            synthetic += 1
    print(f"  store provenance                 {out.provenance.value}")
    print(f"  events carrying a provenance block {len(events) - missing}/{len(events)}")
    print(f"  events marked SYNTHETIC_TEST       {synthetic}")
    assert synthetic == 0, "a synthetic event reached a REAL_REPLAY projection"

    # ---------------------------------------------------------- features
    start_ts = config_payload["start_ts"]
    end_ts = config_payload["end_ts"]
    counts: collections.Counter = collections.Counter()
    first_valid = None
    t = 0.0
    while start_ts + t <= end_ts:
        fv = compute(events, round_id, start_ts + t, result.p0, FeatureConfig())
        if isinstance(fv, FeatureVector):
            counts["VALID"] += 1
            first_valid = first_valid or fv
        else:
            counts[fv.reason.value] += 1
        t += args.step

    print("\nCAUSAL FEATURE VECTORS FROM REAL DATA")
    print("-" * 78)
    for name, n in counts.most_common():
        print(f"  {name:<34} {n:>5}")
    if first_valid is not None:
        f = first_valid
        print(f"\n  sample @ t={f.t:.0f}s tau={f.tau:.0f}s")
        print(f"    twap={f.twap:.2f}  spot={f.spot:.2f}  clob_mid={f.clob_mid:.3f}")
        print(f"    gap_twap={f.gap_twap_bp:.2f}bp  gap_spot={f.gap_spot_bp:.2f}bp  "
              f"lead_gap={f.lead_gap_bp:.2f}bp")
        print(f"    realized_vol={f.realized_vol:.6f}  z_gap={f.z_gap:.3f}  spread={f.spread:.3f}")

    print("\nVERDICT")
    print("-" * 78)
    ok = counts["VALID"] > 0 and synthetic == 0
    print(f"  one captured real round produces causal FeatureVectors: {counts['VALID'] > 0}")
    print(f"  zero synthetic events in that round:                    {synthetic == 0}")
    print(f"  MOS/tick read from market metadata:                     "
          f"{constraints.min_order_shares} / {constraints.tick_size}")
    print("  no real orders were sent (this script has no order path)")
    print("=" * 78)

    out.close()
    raw.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
