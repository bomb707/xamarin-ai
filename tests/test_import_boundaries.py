"""Phase 12C.1 item 6: the permanent production import boundary.

This is the CI guard behind the phase's headline invariant:

    Production/live code can never accidentally consume synthetic data.

Comments and naming conventions do not enforce that; an AST-level assertion
over every shipped module does. The repository has no `.github/workflows`,
so `pytest` IS CI and this file is where the guard lives.

The checks are structural on purpose. They fail on an `import` statement, not
on a code path being exercised, so a violation is caught the moment someone
writes it rather than whenever a particular branch happens to run.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "xamarinbot"

#: Module prefixes that fabricate data or exist only for development. Nothing
#: shipped inside `src/xamarinbot/` may import any of them.
FORBIDDEN_PREFIXES = ("xamarinbot.synthetic", "devtools", "tests")

def real_scripts() -> list[pathlib.Path]:
    """Every TOP-LEVEL script in `scripts/`, discovered automatically.

    Phase 12C.2 item 5: this was a hand-maintained tuple that had already
    fallen behind - `run_real_replay_smoke.py` and `write_capture_manifest.py`
    were added in 12C.1 and silently escaped the guard. A manually curated
    allowlist of things to check is exactly the wrong shape: the failure mode
    is forgetting to add one, and forgetting produces silence.

    `scripts/dev_synthetic/**` is excluded by construction - those are the
    synthetic demos, and importing the generator is their whole purpose.
    """
    return sorted(
        p for p in (REPO_ROOT / "scripts").glob("*.py")
        if "__pycache__" not in p.parts
    )

#: Deleted in Phase 12C.1 item 7 as duplicate/superseded real-market sources,
#: or deprecated pending Phase 13. Re-introducing an import of any of these
#: would recreate "two apparent production sources for the same market datum".
RETIRED_MODULES = (
    "xamarinbot.feeds.chainlink_twap",
    "xamarinbot.feeds.spot_composite",
    "xamarinbot.feeds.polymarket_clob",
    "xamarinbot.feeds.polymarket_user",
)


def python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name this file imports, as written."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import - cannot escape the package
                continue
            if node.module:
                out.add(node.module)
    return out


def violates(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def referenced_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


# ------------------------------------------------------------- the guard

def test_no_shipped_module_imports_fabricated_data_machinery():
    """The whole `src/xamarinbot/**` tree, not just realtime/ and shadow/.

    The stronger form is achievable because Phase 12C.1 replaced the one
    genuine dependency: `walkforward/pipeline.py` and `model/dataset.py`
    imported `SyntheticRoundResult` from the generator purely as a typed
    carrier, and now import the neutral `xamarinbot.rounds.RoundLabel`.
    """
    offenders = []
    for path in python_files(SRC):
        for module in imported_modules(path):
            if violates(module, FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert not offenders, (
        "shipped code must never import fabricated-data or dev-only machinery:\n  "
        + "\n  ".join(offenders)
    )


def test_real_market_scripts_import_no_fabricated_data():
    scripts = real_scripts()
    assert scripts, "scripts/ must contain at least one real-market entry point"
    offenders = []
    for path in scripts:
        for module in imported_modules(path):
            if violates(module, FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert not offenders, "real-market scripts must never import synthetic data:\n  " + "\n  ".join(offenders)


def test_the_guard_covers_every_top_level_script():
    """The discovery itself is asserted, so a new top-level script cannot be
    added without the guard picking it up."""
    discovered = {p.name for p in real_scripts()}
    on_disk = {p.name for p in (REPO_ROOT / "scripts").glob("*.py")}
    assert discovered == on_disk
    assert not any((REPO_ROOT / "scripts" / "dev_synthetic" / n).exists() for n in discovered)


def test_no_module_anywhere_references_a_mock_adapter():
    """Item 6: "production real-market service modules must not import a
    `Mock*` adapter". Phase 12C.1 item 3 renamed the replay adapters, so the
    correct count of `Mock*` names in shipped code is now zero - the replay
    layer replays REAL captured data and must not be named as though it
    fabricates."""
    offenders = []
    for path in python_files(SRC):
        for name in referenced_names(path):
            if name.startswith("Mock"):
                offenders.append(f"{path.relative_to(REPO_ROOT)} references {name}")
    assert not offenders, (
        "shipped code must contain no Mock* adapter references:\n  " + "\n  ".join(offenders)
    )


def test_retired_and_deprecated_adapters_have_no_importers():
    """Item 7: one canonical live-market implementation."""
    searched = python_files(SRC) + real_scripts()
    offenders = []
    for path in searched:
        for module in imported_modules(path):
            if module in RETIRED_MODULES:
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports retired {module}")
    assert not offenders, "\n  ".join(offenders)


def test_superseded_adapter_files_are_gone():
    for gone in ("chainlink_twap.py", "spot_composite.py", "polymarket_clob.py"):
        assert not (SRC / "feeds" / gone).exists(), (
            f"feeds/{gone} was superseded by realtime/* and must not return"
        )
    assert (SRC / "realtime" / "feed_adapter.py").exists(), (
        "the canonical real-market Phase-1 adapter must live under realtime/"
    )


def test_synthetic_generator_is_outside_the_shipped_package():
    assert not (SRC / "synthetic").exists(), (
        "the data fabricator must not live in the production runtime namespace"
    )
    assert (REPO_ROOT / "devtools" / "synthetic" / "rounds.py").exists()


def test_synthetic_demo_scripts_are_unmistakable():
    """Item 5: it must not be possible to mistake a synthetic demo for a real
    run. Every demo lives under scripts/dev_synthetic/ and is named
    `run_synthetic_*`; scripts/ itself holds only real-market entry points."""
    dev_dir = REPO_ROOT / "scripts" / "dev_synthetic"
    assert dev_dir.is_dir()
    for path in dev_dir.glob("*.py"):
        assert path.name.startswith("run_synthetic_"), (
            f"{path.name} generates synthetic rounds and must be named run_synthetic_*"
        )

    top_level = {p.name for p in (REPO_ROOT / "scripts").glob("*.py")}
    assert top_level == {
        "run_market_discovery.py",
        "run_real_recorder.py",
        "run_continuous_capture.py",
        "resolve_capture_labels.py",
        "run_real_replay_smoke.py",
        "write_capture_manifest.py",
        "reindex_captures.py",
        "run_real_shadow.py",
    }, f"scripts/ must hold only real-market entry points; found {sorted(top_level)}"


@pytest.mark.parametrize("rel", [str(p.relative_to(REPO_ROOT)) for p in real_scripts()])
def test_real_scripts_place_no_orders(rel):
    """Item 14 / 18: no real orders are sent from any real entry point."""
    text = (REPO_ROOT / rel).read_text().lower()
    for banned in ("post_order", "create_order", "sign_order", "private_key", "eip712"):
        assert banned not in text, f"{rel} references order-placing symbol {banned!r}"
