import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Phase 12C.1 item 5: the synthetic demos moved to scripts/dev_synthetic/ and
# were renamed `run_synthetic_*` so a reader can never mistake one for a real
# run. `tests/test_replay_determinism.py` imports one of them directly to
# assert the replay pipeline is deterministic, so that directory (not
# `scripts/`, which now holds only real-market entry points) goes on the path.
for extra in (REPO_ROOT / "scripts" / "dev_synthetic", REPO_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
