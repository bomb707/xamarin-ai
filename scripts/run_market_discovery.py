#!/usr/bin/env python3
"""Phase 12C item 2: discover the live BTC 5-minute market and dump every
metadata field the recorder will persist.

Read-only. Places no orders and needs no credentials.

    python scripts/run_market_discovery.py [--rounds N] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from xamarinbot.realtime.discovery import (  # noqa: E402
    MarketDiscovery,
    MarketDiscoveryError,
    ROUND_SECONDS,
    round_start_for,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=2, help="how many consecutive rounds to inspect")
    ap.add_argument("--offset", type=int, default=0, help="rounds to skip forward from the current one")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    discovery = MarketDiscovery()
    base = round_start_for(time.time()) + args.offset * ROUND_SECONDS
    out = []
    for i in range(args.rounds):
        start = base + i * ROUND_SECONDS
        try:
            m = discovery.discover_round(start)
        except MarketDiscoveryError as exc:
            print(f"[{start}] discovery failed: {exc}", file=sys.stderr)
            continue
        if args.json:
            out.append({
                "round_id": m.round_id, "condition_id": m.condition_id,
                "question_id": m.question_id, "slug": m.slug, "question": m.question,
                "resolution_source": m.resolution_source,
                "start_ts": m.start_ts, "end_ts": m.end_ts,
                "up_token_id": m.up_token_id, "down_token_id": m.down_token_id,
                "tick_size": m.tick_size, "min_order_size": m.min_order_size,
                "fees": m.fees.raw, "effective_fee_rate": m.fees.effective_rate,
                "taker_delay_ms": m.taker_delay_ms,
                "settlement_kind": m.settlement_kind, "twap_window_s": m.twap_window_s,
                "outcome_label_source": m.outcome_label_source,
                "is_executable": m.is_executable, "warnings": list(m.warnings),
            })
            continue
        fmt = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(m.start_ts))
        print("=" * 72)
        print(f"round_id           {m.round_id}")
        print(f"question           {m.question}")
        print(f"market / event id  {m.market_id} / {m.event_id}")
        print(f"condition_id       {m.condition_id}")
        print(f"question_id        {m.question_id}")
        print(f"window             {fmt}  +{m.duration_s:.0f}s")
        print(f"resolution source  {m.resolution_source}")
        print(f"settlement kind    {m.settlement_kind}  (twap window {m.twap_window_s}s)")
        print(f"UP   token         {m.up_token_id}")
        print(f"DOWN token         {m.down_token_id}")
        print(f"outcome labels via {m.outcome_label_source}")
        print(f"tick / min size    {m.tick_size} / {m.min_order_size}")
        print(f"fee rate           {m.fees.effective_rate}  type={m.fees.fee_type} "
              f"maker/taker base={m.fees.maker_base_fee}/{m.fees.taker_base_fee}")
        print(f"taker delay        {m.taker_delay_ms} ms")
        print(f"executable now     {m.is_executable}")
        for w in m.warnings:
            print(f"  ! {w}")
        print(f"rules/description  {m.description[:200]}...")
    if args.json:
        print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
