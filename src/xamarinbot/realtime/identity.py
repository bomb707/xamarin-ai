"""Which code produced this capture (Gate A.0.1 item 6).

The problem this closes
-----------------------
Gate A.0 changed how parse failures, labels and eligibility are recorded,
but the continuous recorder was a long-lived process started BEFORE that
commit. Python loads modules once, so the running process kept executing the
old code while the repository showed the new code - and every round it
captured looked, from the outside, like a Gate A.0 round.

There was no way to tell from a capture which code wrote it. That is the
actual defect: not the stale process, but the fact that a stale process
leaves no trace.

Every recorder session now stamps its own identity into the capture, so a
round can always be traced to the exact code that produced it, and captures
from different recorder generations can be reported separately instead of
being silently pooled.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: Bumped whenever the recorder changes what it writes or what it means.
#: 1 = Phase 12C .. Gate A.0; 2 = Gate A.0.1 (structured failure attribution,
#: session identity).
RECORDER_SCHEMA_VERSION = 2

#: Captures with no session identity at all. They were written by a process
#: that predates this module, so their generation is known only negatively.
LEGACY_RECORDER = "LEGACY_RECORDER"
POST_A0_1_RECORDER = "POST_A0_1_RECORDER"


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass(frozen=True)
class RecorderIdentity:
    """The identity of one recorder process.

    `code_sha` is the commit checked out when the process STARTED, which is
    what the process actually loaded - not what HEAD says later. `code_dirty`
    matters just as much: a clean SHA on a dirty tree would be a false
    provenance claim, since the loaded code is then not any commit.
    """

    recorder_code_sha: str | None
    recorder_code_dirty: bool
    process_pid: int
    process_started_at: float
    python_version: str
    recorder_schema_version: int
    recorder_generation: str
    host: str = ""

    @classmethod
    def capture(cls, repo_root: Path | str | None = None) -> "RecorderIdentity":
        """Snapshot the identity of the CURRENT process. Call once at startup."""
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        sha = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain")
        return cls(
            recorder_code_sha=sha,
            recorder_code_dirty=bool(status),
            process_pid=os.getpid(),
            process_started_at=time.time(),
            python_version=sys.version.split()[0],
            recorder_schema_version=RECORDER_SCHEMA_VERSION,
            recorder_generation=POST_A0_1_RECORDER,
            host=platform.node(),
        )

    def as_dict(self) -> dict:
        return {
            "recorder_code_sha": self.recorder_code_sha,
            "recorder_code_dirty": self.recorder_code_dirty,
            "process_pid": self.process_pid,
            "process_started_at": self.process_started_at,
            "python_version": self.python_version,
            "recorder_schema_version": self.recorder_schema_version,
            "recorder_generation": self.recorder_generation,
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "RecorderIdentity":
        return cls(
            recorder_code_sha=row.get("recorder_code_sha"),
            recorder_code_dirty=bool(row.get("recorder_code_dirty")),
            process_pid=int(row.get("process_pid") or 0),
            process_started_at=float(row.get("process_started_at") or 0.0),
            python_version=row.get("python_version") or "",
            recorder_schema_version=int(row.get("recorder_schema_version") or 0),
            recorder_generation=row.get("recorder_generation") or LEGACY_RECORDER,
            host=row.get("host") or "",
        )


def legacy_identity() -> RecorderIdentity:
    """The identity of a capture that carries none.

    Deliberately not `None`: downstream reporting must be able to name the
    generation of every round, and "we do not know" is itself a finding.
    Such captures are not deleted - they remain recoverable through
    revalidation, they simply cannot claim to have been produced by the
    current code.
    """
    return RecorderIdentity(
        recorder_code_sha=None,
        recorder_code_dirty=False,
        process_pid=0,
        process_started_at=0.0,
        python_version="",
        recorder_schema_version=1,
        recorder_generation=LEGACY_RECORDER,
    )
