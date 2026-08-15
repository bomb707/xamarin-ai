# Phase 12B Audit

Audit of `main` against `Xamarinbot_V2_Phase12B_Claude_Prompt.pdf`'s 38
numbered items. **This is an audit only — no implementation changes were
made in this pass**, per the prompt's explicit instruction ("Start with
the audit only... WAIT FOR APPROVAL AFTER THAT AUDIT BEFORE DOING THE
LARGE IMPLEMENTATION PASS").

Every item below was checked against the actual current source (not
against memory of what was intended) — file paths, function names, and
quoted code are all read directly from `main` as of this audit.

Classification key: **CONFIRMED BUG** / **CONFIRMED DESIGN LIMITATION** /
**INTENTIONAL BUT UNSUITABLE FOR PRODUCTION** / **NOT A PROBLEM** /
**NEEDS REAL DATA TO DETERMINE**.

**Revision note (round 2)**: a reviewer pass over this audit found
several additional high-severity issues (train/eval leakage, no
per-window model retraining, uncalibrated `q`, a second baseline harness
bug, and an execution-path inconsistency), plus a real mathematical error
in this document's own proposed fix for items 12-13. All of it verified
against actual code below in **Addendum A-L**, appended after the
original 38-item audit and before the Summary section (which has been
updated accordingly). Item 12-13's original "Proposed fix" text below is
kept as written for the audit trail, but is **superseded** by Addendum F
— see the note inserted at that section. **Still no implementation
changes have been made — this remains audit-only.**

---

## 0. Project objective correction

**Finding**: `grep -rn "profit_target\|stop_after_profit" src/ scripts/ tests/ docs/` returns
zero matches anywhere in the repository. No fixed-dollar profit-target,
stop-after-profit, or equivalent logic was ever implemented in this build
— the "$3-$10 per round" language does not appear in any source file,
config default, or doc.

**Classification**: **NOT A PROBLEM** (nothing to fix — whatever earlier
instruction this section is correcting was not carried into this repo).
Noted so it's on record that this was checked, not assumed.

The stated objective (`maximize E[long-run net trading PnL]` after
costs, WAIT always valid, no per-round PnL requirement) is consistent
with how `OneStepController.decide()` is actually built: `WAIT` is
always a valid, always-present candidate (`wait_candidate()`,
[optimizer/candidates.py:273](../src/xamarinbot/optimizer/candidates.py#L273)),
and candidate selection is `argmax` over `ev_after` (+ `lambda_g * g_after`),
never a fixed target comparison.

---

## 1. Exact portfolio kernel

**File**: [portfolio/state.py](../src/xamarinbot/portfolio/state.py), [portfolio/math.py](../src/xamarinbot/portfolio/math.py)

**Current behavior**: `Pi_U = U - C`, `Pi_D = D - C`, `G = min(Pi_U, Pi_D)`,
`R = Pi_U - Pi_D` are implemented exactly as given, and covered by
`tests/test_portfolio_math.py` with Hypothesis property tests asserting
these identities hold after arbitrary randomized multi-fill sequences.
`EV(P) = q*Pi_U + (1-q)*Pi_D` is not stored as a single named field but
is exactly what `evaluate_taker_candidate`/`evaluate_maker_candidate`
compute via `q*delta_U + (1-q)*delta_D - delta_C` (see item 5 — this is
the *delta* form of the same identity, verified algebraically below).

**Classification**: **NOT A PROBLEM**. No genuine implementation error
found. Per the prompt's own instruction, this infrastructure is being
preserved, not rewritten.

---

## 2. Audit deliverable

This document. Delivered before any Phase 12B code changes.

---

## 3. Baseline `spot_prev` bug (walk-forward harness)

**File**: [walkforward/ablations.py:154](../src/xamarinbot/walkforward/ablations.py#L154), function `_run_baseline_round()`

**Current behavior** (verified by direct read):
```python
inputs = BaselineInputs(
    t=decision_ts, p0=p0, twap=twap_obs.value, clob_mid=mid, clob_mid_prev=mid_prev,
    spot=spot_obs.value, spot_prev=spot_obs.value,   # <-- identical value passed twice
    ...
)
```
`spot` and `spot_prev` are the *same* `spot_obs.value` at every decision
point. `baseline/strategy.py:87` then computes
`spot_direction = _sign(inputs.spot - inputs.spot_prev)`, which is
`_sign(0.0) = 0` **unconditionally, every decision, every round.**

**Root cause confirmed further**: `BaselineConfig.spot_lookback_s = 3.0`
already exists as a config field and is completely unused in this
function. The file already implements the *correct* pattern one field
over — `clob_mid_prev` is built from a genuine second cursor
(`prev_cursor`/`prev_book_feed`) advanced to `decision_ts - cfg.clob_lookback_s`
— it just was never applied to spot.

**This is isolated to the Phase 11 harness, not the baseline logic
itself**: `baseline/strategy.py::decide()` is correct and is exercised
with genuinely different `spot`/`spot_prev` values in
`tests/test_baseline_strategy.py` (e.g. line 27: `spot_prev=100_000.0` vs
`spot=100_100.0`), and Phase 0's own original demo script,
`scripts/run_baseline_replay.py:146-147`, already builds the lookback
correctly:
```python
spot_prev_payload = spot_index.latest_payload_before(decision_time - cfg.spot_lookback_s)
spot_prev = spot_prev_payload["value"] if spot_prev_payload else spot_obs.value
```
So this is a regression introduced specifically when Phase 11 rebuilt a
parallel baseline runner instead of reusing Phase 0's already-correct one.

**Mathematical consequence**: `spot_direction` is identically 0 (`FLAT`
in `_sign()` terms) at every single baseline decision point, for every
round, in every Phase 11/demo run that used `_run_baseline_round`.

**Trading consequence**: Since the baseline requires
`clob_direction == spot_direction == twap_direction` (unanimous), and
`twap_direction` is only ever exactly 0 when the gap is below
`minimum_gap_bp` (rare), unanimity with `spot_direction` pinned at 0 is
**never achievable** except in the degenerate case where `twap_direction`
is *also* 0 (in which case the round is already skipped via
`GAP_BELOW_MINIMUM` before the unanimity check even runs). **The
baseline is structurally incapable of ever placing a trade through this
harness.** This fully explains why every "1_baseline_unanimous" result in
every Phase 11 ablation run and the walk-forward demo showed
`n_actions=0, pnl=0.000` — a result I reported without flagging as
suspicious in the prior session. It should have been flagged.

**Trading consequence, broader**: **every Phase 11 "baseline vs V2"
comparison in `docs/PHASE_STATUS.md` and every ablation-matrix run to
date is invalid** — the baseline arm was never actually a live comparison
point, it was a guaranteed zero.

**Classification**: **CONFIRMED BUG**. High severity — invalidates a
whole ablation arm across every Phase 11 run performed so far.

**Proposed fix**: Mirror the existing `prev_cursor`/`prev_book_feed`
pattern already in the same function: add a `MockSpotFeed(prev_cursor)`,
advance `prev_cursor` to `decision_ts - cfg.spot_lookback_s` (already
done for the CLOB cursor at that same line), and pass its `.value` as
`spot_prev`. (Two separate `prev_cursor`s at different lookback horizons
would be needed since `clob_lookback_s` and `spot_lookback_s` can differ
— currently both default to 3.0s but are independent config fields.)

**Regression test required**: A rising historical spot sequence must
give `spot_direction = UP` and a falling one `spot_direction = DOWN`
*through the ablations harness specifically* (not just through
`baseline/strategy.py::decide()` directly, which is already covered) —
i.e. a test that constructs a round with a clear spot trend and asserts
`_run_baseline_round` (or its post-fix equivalent) produces at least one
non-zero-`spot_direction` decision.

---

## 4. Synthetic data is not profitability evidence

**File**: [synthetic/rounds.py:122-128](../src/xamarinbot/synthetic/rounds.py#L122)

**Current behavior** (verified):
```python
gap_bp = 10_000.0 * (spot_val - twap_val) / twap_val
mid = 1.0 / (1.0 + math.exp(-0.001 * gap_bp)) + rng.gauss(0.0, 0.002)
```
The synthetic CLOB midpoint is constructed as a direct (noisy) sigmoid of
the *same* spot-vs-TWAP gap that also drives the predictive features
(`Z_gap`, lead-lag) **and** the settlement outcome
(`outcome = Side.UP if final_twap > p0 else Side.DOWN`, line ~202).

**Mathematical/trading consequence**: CLOB direction, spot direction,
TWAP direction, and the settlement outcome are all mechanically
correlated by construction, not independently causal. A model "learning"
from `Z_gap` on this dataset is partly recovering the exact signal that
wrote the mid-price and the outcome, not discovering real market
structure. This is a more precise root-cause statement of something
already partially observed and documented in `docs/PHASE_STATUS.md`'s
"Phase 5 demo finding" (combined lead-lag model failing to beat
TWAP-only, attributed there to "gap_twap_bp is what the settlement
outcome is actually keyed off of") — the audit correctly identifies the
generator mechanism behind that symptom.

**Classification**: **CONFIRMED DESIGN LIMITATION** (already partially
self-documented as a symptom; this audit pins down the mechanism).
`synthetic/` remains appropriate for unit/regression/causality/failure-
injection tests (its stated purpose) but not for *any* profitability,
parameter-optimality, or "X beats Y" claim. Every such claim currently in
`docs/PHASE_STATUS.md` (Phase 5, 9, 10, 11, 12 "demo findings") is
correctly caveated there already as synthetic-only, but this audit
confirms those caveats are load-bearing, not boilerplate.

**Proposed fix**: No code fix — this is a statement about what synthetic
data can and cannot prove. Action item: audit `docs/PHASE_STATUS.md` for
any sentence that could be read as an uncaveated profitability claim and
tighten wording; more importantly, do not let any *future* PR state a
tuned-parameter or "beats baseline" claim without the synthetic caveat.

**Regression test required**: N/A (documentation/process, not code).

---

## 5. `ev_after` naming vs. semantics

**File**: [optimizer/candidates.py:104](../src/xamarinbot/optimizer/candidates.py#L104), function `_finalize()`

**Current behavior**:
```python
ev_after_raw = q * delta_U + (1.0 - q) * delta_D - delta_C   # (in evaluate_taker_candidate / evaluate_maker_candidate)
ev_after = ev_after_raw - cfg.churn_penalty                   # in _finalize()
```
This is `q*ΔU + (1-q)*ΔD - ΔC`, a function of the fill's *deltas* only.

**Mathematical check (the identity the prompt asks for)**: Let
`EV(state) = q*U + (1-q)*D - C`. Then
`EV(after) - EV(before) = q*(U_after-U_before) + (1-q)*(D_after-D_before) - (C_after-C_before)`
`= q*delta_U + (1-q)*delta_D - delta_C`
— exactly the current formula, **by linearity**, since the same `q` is
used on both sides. So the current field is mathematically identical to
`EV_after_total - EV_before_total`, i.e. it **is** `ΔEV` under the exact
name `ev_after`. The prompt's claim that this "is not necessarily the
total expected PnL of the portfolio after the order" is true of the
*name* but the *value* is provably the correct delta — this is not a
math bug, every consumer (`argmax` selection against a common baseline
state) already uses it correctly as an incremental value.

**Classification**: **CONFIRMED DESIGN LIMITATION** (naming/documentation,
not a math bug). The field should be renamed `delta_ev` (or the identity
documented inline) so a future reader doesn't misread it as total
portfolio EV and, e.g., try to sum it across a portfolio history to get
a running total (which delta values *can* validly do, but only if every
reader knows that's what's happening).

**Proposed fix**: Rename `ev_after` → `delta_ev` throughout
`CandidateAction`, `_finalize`, both `evaluate_*_candidate` functions,
the controller's `argmax` key, journal schema, and every report/test that
reads `.ev_after` — a mechanical but wide-reaching rename (used in
~15 files by grep). Add a docstring proving the identity above so it's
never re-litigated.

**Regression test required**: `EV_after_total - EV_before_total == delta_ev`
computed independently via `q*U_after+(1-q)*D_after-C_after` minus
`q*U_before+(1-q)*D_before-C_before`, for a random sample of
portfolio/candidate combinations (Hypothesis-style, matching the existing
property-test style in `test_portfolio_math.py`).

---

## 6. Marginal edge vs. total candidate EV

**File**: [optimizer/candidates.py:111](../src/xamarinbot/optimizer/candidates.py#L111) (`_finalize`), [optimizer/config.py:19](../src/xamarinbot/optimizer/config.py#L19) (`edge_min`)

**Current behavior**:
```python
if apply_edge_min and purpose is not OrderPurpose.HEDGE and ev_after < cfg.edge_min:
    violated.append("edge_min")
```
`edge_min` is compared directly against `ev_after`, a **total dollar**
value that scales with quantity. No `e_U(x) = [q*x - K_U(x)] / x`
per-share marginal-edge concept exists anywhere in the codebase (grepped
`optimizer/`, `portfolio/`, `execution/` — no `marginal` or `e_U`/`e_D`
function). This is a self-acknowledged gap already: `docs/PHASE_STATUS.md`'s
"Known reconstruction gaps" item #14 already states *"SS21 'edge_min:
minimum predicted net edge | Alpha'. Not formularized beyond the name in
the source docs; this build applies it as a flat floor on ev_after"* —
so the existence of the gap was known; this audit adds the concrete
failure mode the prompt names ("a 500-share trade with tiny edge can pass
while a 5-share trade with excellent edge can fail").

**Classification**: **CONFIRMED DESIGN LIMITATION** (previously
self-documented as a naming/formalization gap; now confirmed with the
specific size-vs-quality confound it causes).

**Proposed fix**: Add `min_marginal_edge` (checked against
`e_U(x)`/`e_D(x)` at the *requested* quantity — or, more precisely, at
the marginal/last-unit edge once item 7's marginal-quantity-boundary
optimization exists) as a separate field from the existing
`edge_min` (kept, renamed `min_total_delta_ev`, as an optional
operational floor). Do not tune either against Phase 11/12 synthetic
results (per item 37).

**Regression test required**: A candidate with large quantity and tiny
per-share edge must be rejected by `min_marginal_edge` even though its
total `delta_ev` clears a low `min_total_delta_ev`; a small candidate
with excellent per-share edge must pass `min_marginal_edge` even with
small total `delta_ev`.

---

## 7. Taker candidate sizing from raw book depth

**File**: [optimizer/candidates.py:41](../src/xamarinbot/optimizer/candidates.py#L41), function `taker_quantities()`

**Current behavior**:
```python
def taker_quantities(levels, max_levels):
    out = []
    cumulative = 0.0
    for level in levels[:max_levels]:
        cumulative += level.size
        out.append(cumulative)
    return out
```
Candidates are *only* the cumulative sizes at each of the first
`max_taker_depth_levels` (default 3) book levels — e.g. `[500, 1250, 2250]`
for a book with levels `500/750/1000`. There is no mechanism to generate
an intermediate size (the prompt's "7.4 shares"), no minimum-exchange-size
step, no marginal-edge-boundary size, and — despite
`portfolio/math.py::max_directional_spend(g_current, g_min)` already
existing and being unit-tested in isolation — **it is never called from
`taker_quantities` or anywhere in the candidate-generation path** (grep
confirms zero call sites outside its own test file), so no risk-budget-
derived quantity is ever generated either.

**This is not theoretical — it was already observed and diagnosed in
Phase 11's own audit trail, just not previously connected to this root
cause.** `docs/PHASE_STATUS.md`'s Phase 11 finding for ablation #6
(taker-only) states: *"this dataset's taker candidates are sized from
real order-book depth ... which runs large enough that every one of them
breaches the g_min=-100 risk floor ... while maker's small fixed clip
(maker_quantity=20) comfortably fits."* That is exactly this bug's
consequence, already caught empirically without being traced to
`taker_quantities`'s lack of intermediate sizing at the time.

**Mathematical consequence**: The candidate set for taker execution is
sparse and depth-anchored rather than risk/edge-anchored — the optimizer
can only choose from 3 (or however many) large, level-boundary-determined
sizes, never the actual EV/risk-optimal size in between.

**Trading consequence**: Under any risk floor tighter than the first
book level's depth, taker execution has *nothing feasible to offer* —
demonstrated concretely by ablation #6 showing zero actions despite
having real edge available (confirmed via a standalone trace at the
time).

**Classification**: **CONFIRMED BUG** (elevated from "design limitation"
given the already-observed, reproducible behavioral consequence).

**Proposed fix**: Generate candidate quantities from the union of: (1)
`min_order_size` (once real market metadata is wired, item 22), (2) a
configurable step size, (3) the depth-level boundaries (kept, as an
upper-bound/no-worse-than-today option), (4) the marginal-edge boundary
(largest `x` where `e_U(x) >= min_marginal_edge`, once item 6 exists),
(5) `max_directional_spend(g_current, g_min)` (already implemented,
currently dead code), (6) `position_limit - current_position`, (7) the
per-order/per-round spend caps (item 9). Then evaluate and pick the best
by the optimizer's normal `argmax`, rather than assuming depth levels are
the only relevant boundaries.

**Regression test required**: With a `g_min` tighter than the first book
level's depth-implied cost but looser than some smaller quantity, assert
a valid (non-empty, passing) taker candidate is generated at that smaller
quantity — i.e. a direct regression test reproducing (and then fixing)
the exact ablation #6 scenario already on record.

---

## 8. `limit_price=1.0` as taker "protection"

**File**: [optimizer/controller.py](../src/xamarinbot/optimizer/controller.py) (`evaluate_taker_candidate(..., limit_price=1.0, ...)` for every ALPHA taker candidate), [optimizer/candidates.py:268](../src/xamarinbot/optimizer/candidates.py#L268) (same for HEDGE candidates)

**Current behavior**: Every taker candidate — alpha and hedge — is
evaluated with `limit_price=1.0` hardcoded, i.e. "accept any price up to
$1.00." `walk_depth` (execution/taker.py) walks book levels up to the
requested quantity or until `limit_price` is hit; since `limit_price=1.0`
is effectively unconstraining (prices are probabilities in `[0,1]`),
there is no real worst-price protection anywhere in this path. No
`p_U^max` derivation from `q`, fees, `min_marginal_edge`, slippage, or
risk exists.

**Also confirmed**: no dollar-amount-based order translation layer exists
anywhere in the codebase (grepped `execution/`, `feeds/polymarket_clob.py`
for "dollarAmount"/"amount" — none found). `execution/taker.py::walk_depth`
and everything downstream operates purely in share quantities.

**Mathematical consequence**: None in the current *backtest* context —
`walk_depth` is evaluated against a single, already-fetched, static book
snapshot, so "accept any price up to 1.0" and "accept only up to a
derived `p_max`" produce identical results when the book itself is fixed
at evaluation time (there's nothing for a 1.0 cap to protect against in a
snapshot). This matters once real order submission is live.

**Trading consequence (real, live-relevant)**: In live trading, a
FAK/FOK order submitted with an effectively-uncapped limit price has no
protection against a thin, wide, or momentarily-adverse book at the
moment of matching (the actual Polymarket order API also takes a dollar
amount, not a share quantity, per the prompt's note — confirmed absent
here too). This is a real gap for Phase 13 live-readiness, not (yet) a
backtest-correctness bug.

**Classification**: **INTENTIONAL BUT UNSUITABLE FOR PRODUCTION** for the
worst-price question (a deliberate backtest simplification that must not
carry into live submission), **CONFIRMED DESIGN LIMITATION** for the
missing shares→dollarAmount→worstPriceLimit translation layer (never
built at all, not a live/backtest split).

**Proposed fix**: Derive `p_U^max` from `q`, fees, `min_marginal_edge`,
and a configured safety margin (roughly `q > c_marginal + safetyMargin`
per share, translated to a worst acceptable average price for the
requested quantity). Build the translation layer
`x*_shares -> dollarAmount* -> worstPriceLimit*` as an execution-layer
concern (not inside `optimizer/`), with fill reconciliation against
actual returned shares afterward. Keep `walk_depth`'s share-based
interface for backtesting; the dollar-amount translation is specifically
a live-order-submission concern.

**Regression test required**: A candidate must be rejected (or capped) if
its fee-inclusive average execution price would exceed the derived
`p_max`, even when raw book depth would otherwise support the full
requested quantity. Separately, a share→dollar→shares round-trip test
for the new translation layer once built.

---

## 9. Round-level `spend_cap` not enforced cumulatively

**File**: [portfolio/math.py:121-123](../src/xamarinbot/portfolio/math.py#L121), function `evaluate_constraints()`

**Current behavior**:
```python
spend = result.delta_C
if constraints.spend_cap is not None and spend > constraints.spend_cap:
    violated.append("spend_cap")
```
`result.delta_C` is `FillSimulationResult.delta_C`, which is
`portfolio_after.C - portfolio_before.C` for **one single candidate
fill** — not `portfolio_after.C - C_at_round_start`. `RiskConstraints`
carries no notion of round-start cost at all.

**Mathematical consequence**: Given `spend_cap=200` and a portfolio that
already spent `$180` this round from prior fills, a *new* candidate
costing `$50` passes this check (`50 <= 200`) even though
`180 + 50 = 230 > 200` — the round-level budget is silently exceeded.
Each order is checked in isolation against the *full* cap, not the
*remaining* cap.

**Trading consequence**: A sequence of individually-legal orders can
overspend the intended round budget by an unbounded multiple, limited
only by how many decision points occur before the round ends. This is a
real risk-control gap, not cosmetic.

**Note on the baseline**: `baseline/strategy.py:147-149` gets this
*right* already — `projected_spend = state_C + quantity * limit_price;
if projected_spend > cfg.spend_cap` correctly compares *cumulative*
projected spend against the cap. The bug is specific to
`portfolio/math.py::evaluate_constraints` (the Phase 8+ optimizer path),
not a repo-wide pattern.

**Classification**: **CONFIRMED BUG**.

**Proposed fix**: Thread a `round_start_C` (or equivalently, the caller
already has `portfolio_before` inside `FillSimulationResult` from the
*start of the round*, not just from the immediately-prior candidate —
this requires the caller to track and pass round-start state explicitly,
since `evaluate_constraints` currently only sees one fill's before/after).
Enforce `C_after - C_round_start <= spend_cap`. If both a per-order cap
and a per-round cap are wanted, represent them as two distinct
`RiskConstraints` fields (e.g. `per_order_spend_cap`, `round_spend_cap`).

**Regression test required**: Two sequential candidates, each individually
under `spend_cap`, whose *sum* exceeds it — the second must be rejected
(currently would pass).

---

## 10. Favored-side inference from payoff geometry, not prediction

**File**: [optimizer/candidates.py:69-70](../src/xamarinbot/optimizer/candidates.py#L69), function `_favored_side()`

**Current behavior**:
```python
def _favored_side(portfolio: PortfolioState) -> Side:
    return Side.UP if portfolio.Pi_U >= portfolio.Pi_D else Side.DOWN
```
At `U=D=C=0` (a flat/empty portfolio, `Pi_U == Pi_D == 0.0`), this
returns `UP` purely because of the `>=` tie-break — an arbitrary default,
not a predictive judgment. This function feeds `favored_side` into
`evaluate_constraints`'s `p_min` check only
(`optimizer/candidates.py:100-101`).

**Confirmed scope**: grepped the entire `src/` tree for `favored_side` —
this is the *only* call site; nothing in `supervisor/` or `mpc/` uses an
equivalent inference, so this is a single, contained instance, not a
repo-wide pattern.

**Also confirmed currently dormant**: `OneStepConfig.p_min` defaults to
`None`, and no ablation, demo, or test in the repo ever sets `p_min` to a
real value — `favored = _favored_side(portfolio) if cfg.p_min is not None
else None` means this function is **never actually called with an effect
on any existing result to date**. The bug is real but has not yet
corrupted any reported number, because the feature it feeds is unused.

**Mathematical consequence**: If/when `p_min` is ever configured, the
`p_min` floor would apply to a side chosen by portfolio geometry, not by
which side actually has the better executable edge — potentially
constraining the *wrong* side's profit floor relative to what the
predictive model and current book actually favor.

**Classification**: **CONFIRMED BUG** (currently dormant/inactive, but
would misbehave immediately upon `p_min` being configured).

**Proposed fix**: Compute `Side* = argmax(DeltaJ_U, DeltaJ_D)` from
current executable opportunity (the marginal edge functions from item 6,
or at minimum `q` vs `1-q` net of near-touch cost) and pass it explicitly
wherever a favored-side constraint is evaluated, replacing
`_favored_side`'s payoff-geometry heuristic.

**Regression test required**: At a flat portfolio (`Pi_U == Pi_D == 0`)
with `q` clearly favoring DOWN (e.g. `q=0.2`) and a real `p_min`
configured, the favored side used for the `p_min` check must be DOWN, not
the current UP default.

---

## 11. `g_min` hardcoded, not bankroll-relative

**File**: [optimizer/config.py:11](../src/xamarinbot/optimizer/config.py#L11) (`g_min: float`, no default — a required field), every call site (`ablations.py:64`, all `scripts/run_*_demo.py`)

**Current behavior**: `g_min` is a required field with no default, and
every caller in the repo passes a literal like `-100.0` or `-200.0`.
Grepped every config construction site — no bankroll, drawdown-state,
calibrated-confidence, volatility, `tau`, or liquidity input feeds into
any of them. `docs/PHASE_STATUS.md` already documents that `-100.0` (and
the various `lambda_g`, `churn_penalty` values tuned alongside it) were
chosen empirically by *matching this specific synthetic dataset's typical
G-after magnitude distribution* — which is itself proof that these values
were never derived from a bankroll-relative risk formula.

**Mathematical/trading consequence**: The risk floor has no principled
connection to actual capital at risk. A `g_min=-100` is meaningless
without knowing the bankroll it's a fraction of.

**Classification**: **CONFIRMED DESIGN LIMITATION** (already implicitly
acknowledged via the "empirically tuned to this dataset" language in
`docs/PHASE_STATUS.md`, now stated directly: no `g_min` value used to
date should be read as inherently optimal or even reasonable outside this
specific synthetic dataset).

**Proposed fix**: Per the prompt's own instruction, do *not* invent the
final production formula yet. First expose a `RiskConfig` (bankroll,
drawdown state, calibrated confidence, volatility, `tau`, liquidity,
current portfolio) as explicit inputs to a `g_min_t` *computation*
function, separate from the flat `OneStepConfig.g_min` field it replaces
or wraps. Calibrate any weights only against validation data, never test
data (item 32/37), and only once real data exists.

**Regression test required**: A property test that `g_min_t` scales
monotonically with bankroll (larger bankroll -> looser/more-negative
allowed floor, all else equal) once the formula exists — cannot be
written meaningfully before the formula is designed, so this is a
forward-looking test requirement, not one addable today.

---

## 12-13. Portfolio repair is reactive, not proactive; no explicit buffer economics

**File**: [optimizer/candidates.py:198-270](../src/xamarinbot/optimizer/candidates.py#L198), function `generate_hedge_candidate()`; [portfolio/math.py:143-146](../src/xamarinbot/portfolio/math.py#L143), function `min_hedge_quantity()`

**Current behavior**:
```python
def min_hedge_quantity(l_max, pi_down, c_d):
    return max(0.0, (-l_max - pi_down) / (1.0 - c_d))
```
With `l_max = -cfg.g_min` (e.g. `100.0`), `x_min_hedge` is `0` whenever
`pi_down >= -l_max`, i.e. whenever the worst-case side is still *above*
the risk floor. Combined with `generate_hedge_candidate`'s own early
return (`if math.isclose(portfolio.Pi_U, portfolio.Pi_D): return None`)
and its `if x_min <= 0: return None` — **a hedge candidate is only ever
generated once the worst-case outcome is already at or below the
configured floor.** There is no `BUFFER_BUILD` candidate type, and
`OrderPurpose` (`portfolio/math.py:22-24`) only has `ALPHA` and `HEDGE` —
no `BUFFER_BUILD` or `REBALANCE` exists in the enum at all.

**Mathematical consequence**: The economic strategy the prompt describes
("proactively accumulate cheap opposite-side inventory -> build
settlement buffer -> use some of that buffer for controlled directional
residual") is not implemented at all. What exists is purely defensive:
"only act once already in trouble."

**Trading consequence**: `enable_portfolio_repair=True` (Phase 11's
ablations 6-8) never triggers a hedge while G is comfortably above the
floor — confirmed by the same Phase 11 trace evidence already on record
(hedge candidates essentially never appeared in any ablation trace; the
non-wait activity observed there was overwhelmingly `maker_*`/`taker_*`
ALPHA candidates). This is a distinct root cause from item 7's (taker
sizing) — even where taker/maker candidates *were* feasible, no
proactive buffer-building candidate competed alongside them, ever.

**Classification**: **CONFIRMED DESIGN LIMITATION** — this is the single
largest architectural gap found in this audit relative to the strategy
doc's stated design (SS17's buffer/repair economics), not a small
oversight.

**Proposed fix — SUPERSEDED, kept for audit trail only, see Addendum F**:
~~Add `BUFFER_BUILD` and `REBALANCE` to `OrderPurpose`. Implement the
pair-buffer identity directly: `DeltaG_pair(x) = x - K_U(x) - K_D(x)` for
acquiring `x` on the currently cheaper/under-represented side~~ — **this
formula is mathematically wrong for a one-sided purchase.**
`DeltaG_pair(x) = x - K_U(x) - K_D(x)` is only correct when `x` is
acquired on **both** sides simultaneously (`U'=U+x, D'=D+x`). A
single-sided `BUFFER_BUILD` fill (which is what "acquiring x on the
cheaper side" actually describes) must use
`DeltaG_U(x) = min(U+x,D) - min(U,D) - K_U(x)` instead — see Addendum F
for the full correction and proof. The rest of this paragraph's intent
(generate `BUFFER_BUILD` independent of a risk-floor breach, compete on
the same `argmax` footing, evaluate across sequential fills) stands
unchanged; only the formula was wrong.

**Regression test required**: With `G` comfortably above `g_min` but a
cheap opposite-side buffer opportunity available (`K_U(x)+K_D(x) < x` for
some `x`), a `BUFFER_BUILD` candidate must be generated and must be
selectable by the optimizer — currently impossible since no such
candidate type is ever produced. Separately, a `DeltaG_pair` identity
test verifying `DeltaG_pair(x) == x - K_U(x) - K_D(x)` against the
portfolio kernel's own `G` computation after simulating both fills.

---

## 14. Regime matrix as hard veto

**File**: [optimizer/controller.py](../src/xamarinbot/optimizer/controller.py), `OneStepController.decide()`

**Current behavior**: Candidate-generation blocks are gated by structural
`if` statements on `permitted_actions` *before* any candidate object is
created:
```python
if SeedAction.TAKER_UP in permitted_actions and book_up is not None and book_up.asks:
    for qty in taker_quantities(...):
        candidates.append(evaluate_taker_candidate(...))
```
If `SeedAction.TAKER_UP` is not in `permitted_actions` (from
`ActionPermissionMatrix.permitted_actions(classify_seed_action(...))`),
**no `taker_up_*` candidate is ever constructed or evaluated** — the
optimizer's `argmax` never sees it, so it cannot be chosen even if it
would have had excellent EV. This is confirmed for all four directional
families (`TAKER_UP`, `TAKER_DOWN`, `MAKER_UP`, `MAKER_DOWN`); the hedge
candidate (item 12) is the one exception already noted as evaluated
"regardless of the regime's permitted_actions" (a deliberate design
choice documented inline).

**Mathematical/trading consequence**: A coarse 6x3x3-bucket categorical
regime state can suppress an economically attractive candidate that a
finer-grained (continuous `q`, real book) view would have accepted. This
is exactly the mechanism already observed and documented as the root
cause of ablation #6's zero-action result *in combination with* item 7 —
if the permitted regime only allows `MAKER_DOWN`+`WAIT` at a given
decision point, no `TAKER_*` candidate exists to evaluate regardless of
whether taker sizing (item 7) gets fixed.

**Classification**: **CONFIRMED DESIGN LIMITATION** — matches the prompt's
description exactly; this is a real architectural choice (Phase 6's
"candidate family only" design, stated explicitly in that module's own
docstring) that the prompt correctly argues should become a soft
prior/penalty rather than a hard veto, at least as an ablation.

**Proposed fix**: Change regime output to a per-family
prior/penalty/urgency weight applied to `delta_ev` or the selection
score, rather than a presence/absence gate on candidate generation.
Generate all directional families always (subject to the *economic*
constraints — book existence, fee viability), and apply the regime
signal as a continuous adjustment. Keep the current hard-gated behavior
available as an explicit ablation config flag so its effect can be
measured against the soft version (per the prompt's own instruction).

**Regression test required**: With a regime that hard-gates to
`MAKER_DOWN+WAIT` only, but a `TAKER_UP` candidate with excellent EV
available in the book, the soft-prior mode must be able to select
`TAKER_UP` (penalized but not eliminated) while the hard-gated ablation
mode must not.

---

## 15-17. Static maker parameters, maker utility formula, cancellation hysteresis

**File**: [optimizer/config.py:29-31](../src/xamarinbot/optimizer/config.py#L29) (`maker_price_offsets_ticks=(0,1,2)`, `maker_quantity=20.0`, `maker_horizon_s=10.0`, all fixed class defaults); [optimizer/candidates.py:186-189](../src/xamarinbot/optimizer/candidates.py#L186) (maker EV formula); [supervisor/predicates.py:25-28](../src/xamarinbot/supervisor/predicates.py#L25) (`regime_flip`)

**Item 15 — static maker params, confirmed**: `maker_quantity`,
`maker_horizon_s`, and `maker_price_offsets_ticks` are fixed
dataclass-default engineering constants, never derived from marginal
utility, `tau`, volatility, fill hazard, or opportunity decay anywhere in
the codebase (grepped — no function computes a per-decision maker
quantity or TTL; `maker_price_grid` generates candidates from these fixed
offsets, not from an optimized search). **CONFIRMED DESIGN LIMITATION.**

**Item 16 — maker utility formula, checked directly**:
```python
rho = fill_probability(distance_to_touch_ticks, queue_ahead_shares, horizon_s, exec_cfg.maker)
qf = q_fill(q, side, exec_cfg.maker)
ev_if_filled = qf * delta_U + (1.0 - qf) * delta_D - delta_C
ev_after_raw = rho * ev_if_filled - cfg.opportunity_cost
```
The *EV* term (`ev_after_raw`) is correctly `rho`-weighted, matching the
prompt's `rho*[DeltaEV_fill] - OpportunityCost` shape. However, the hard
safety check inside `_finalize`/`evaluate_constraints` uses
`portfolio_after` computed from the **if-filled** portfolio
unconditionally (line 180: `portfolio_after_if_filled = apply_fill(portfolio, fill)`),
i.e. `g_after` (used both for the hard `g_min` gate and for the
`lambda_G * g_after` soft term in the controller's selection score,
`optimizer/controller.py:134`) is the **unweighted, pessimistic if-filled
value**, not `rho`-weighted, while `ev_after` in that same selection sum
*is* `rho`-weighted. **This exact inconsistency was already found and
documented** in `docs/PHASE_STATUS.md`'s Phase 11 `lambda_g` tuning
section: *"maker candidates' ev_after is properly fill-probability-
weighted ... but the g_after used in the SAME score is the pessimistic
if-filled value, unweighted ... an inherent inconsistency between an
expectation term and a conditional-worst-case term added together"* —
confirmed still present, unresolved, and explicitly flagged there as
deferred rather than fixed. The hard `G_ifFill >= G_min` check itself
being conservative (if-filled, unweighted) is correct and should stay
per the prompt's own instruction ("Keep the hard safety test").
**CONFIRMED DESIGN LIMITATION** (already self-documented as deferred).

**Item 17 — cancellation, checked directly**:
```python
def regime_flip(origin_state: RegimeState, current_state: RegimeState) -> bool:
    return origin_state != current_state
```
A pure categorical inequality — any change at all in `RegimeState`
(a single-tick flicker, not necessarily an economically meaningful
change) triggers immediate `CANCEL` via `review_order`'s priority chain
(`risk_breach` -> `regime_flip` -> ...). No hysteresis, debounce, or
persistence mechanism exists. This is confirmed as the direct mechanism
behind the already-documented Phase 9/11/12 finding that maker orders
get cancelled before ever reaching TTL resolution ("56 registered, 55
cancelled via REGIME_FLIP" — Phase 12's own diagnostic, now traced to
this specific one-line predicate). There is also no `V_hold`/`V_cancel`/
`V_replace` economic comparison anywhere — cancellation on `regime_flip`
is unconditional once triggered, never weighed against whether the order
is still economically attractive. **CONFIRMED DESIGN LIMITATION** (a real
architectural gap, and the confirmed root cause of a result already
reported in `docs/PHASE_STATUS.md` under a different framing).

**Proposed fix (15-17 combined)**: Make maker price/qty/TTL a joint
optimization over the boundaries listed in the prompt (best bid,
inside-spread, below-bid-for-oscillation, model-max-price, hedge/buffer
price, tick grid). Fix the `g_after` weighting inconsistency by either
(a) using `rho`-weighted `g_after` in the *soft* `lambda_G` selection
term while keeping the unweighted if-filled value for the *hard*
`g_min` gate (two separate fields), or (b) the fuller
`E[J] = rho*J(fill) + (1-rho)*J(noFill) - J(now)` formula once no-fill
continuation value is well-defined. Replace `regime_flip`'s unconditional
cancel with `action = argmax(V_hold, V_cancel, V_replace)`, keeping hard
cancellation (stale data, invalid tick, hard risk breach, market
closure, invalid state) as unconditional exceptions, and add
hysteresis/persistence (e.g. require N consecutive flipped observations,
or a minimum dwell time in the new state) before a soft regime-change
signal can trigger cancellation.

**Regression test required**: (15) a scenario where the optimal maker
quantity/TTL differs from the fixed defaults and the dynamic version
finds it. (16) a test asserting the two `g_after` values (hard-gate vs
soft-score) are computed consistently once reconciled — currently there
is no test that would catch the inconsistency because both paths use the
same (single) if-filled value throughout, so the test must specifically
assert the *intended* new behavior once implemented. (17) a
UP->WAIT->UP flicker within the hysteresis window must not cancel; a
sustained regime change past the hysteresis threshold must.

---

## 18. Phase 11 supervisor evaluation placeholders

**File**: [walkforward/ablations.py:223](../src/xamarinbot/walkforward/ablations.py#L223), inside `_run_controller_round()`

**Current behavior**:
```python
decision = supervisor.review_order(tracked, decision_ts, snapshot.state, 0.0, portfolio.G, fv.tau, True)
```
Mapped against `review_order`'s signature
(`supervisor/supervisor.py:33-43`): `current_ev_after=0.0` (a **literal
hardcoded placeholder**, never recomputed from the order's actual current
state), `current_g_after_if_fill=portfolio.G` (the portfolio's **current,
unconditional** `G` — not "G if this specific resting order's remaining
shares actually fill," which would require simulating that fill), and
`current_optimal_ev` is **not passed at all** (defaults to `None`, which
means `book_displacement`/`REPLACE` can never fire in this harness at
all, since `book_displacement(...)` is only called when
`current_optimal_ev is not None`).

**Mathematical consequence, checked against `review_order`'s actual
priority chain**:
- `risk_breach(current_g_after_if_fill, cfg)` = `portfolio.G < cfg.g_min`
  — checks the **current** G, not the G that *would result* if this
  order fills. Since filling a resting order typically *worsens* G
  (adds directional exposure), using current G is systematically
  *more optimistic* than the correct if-filled value — a genuine
  under-triggering safety gap, not just a test-validity issue.
- `edge_failure(current_ev_after=0.0, cfg)` = `0.0 < cfg.edge_min` —
  this is a **fixed, input-independent** answer for every order, every
  round, forever (true iff `cfg.edge_min > 0`, false otherwise,
  regardless of the order's real current economics).
- `regime_flip` is unaffected by these placeholders (uses only
  `tracked.origin_regime_state` vs `snapshot.state`, both real) — this
  is why the already-reported "55/56 cancelled via REGIME_FLIP" finding
  is *not* invalidated by this bug (`regime_flip` is checked before
  `edge_failure` in priority order and fires first in the observed
  cases). But `risk_breach` is checked *even earlier* than `regime_flip`
  and is the one most likely to be masking a real trigger.

**Trading consequence**: Phase 11 ablations 7 and 8 (the only two that
enable `use_supervisor=True`) do not actually test dynamic open-order
risk/edge re-evaluation — they test regime-flip cancellation correctly,
but the risk-breach and edge-failure triggers are evaluated against
input-independent or systematically-wrong numbers. On this specific
synthetic dataset the practical effect was likely small (regime_flip
dominates and fires first for most observed cancellations), but this
would matter significantly on real data with slower regime dynamics,
which is exactly the scenario Phase 13 needs to be safe under.

**Classification**: **CONFIRMED BUG**.

**Proposed fix**: Before calling `supervisor.review_order` for each
resting order, recompute from current state exactly as the prompt lists:
remaining shares, current `q`, current `q_fill`, current queue position,
current fill probability, current if-filled cost, current `G` after a
*simulated* fill of the remaining shares (reusing
`evaluate_maker_candidate`'s if-filled-portfolio machinery, not
`portfolio.G` directly), current `ev_after` (via the same evaluation),
and the current optimal candidate's EV (for `book_displacement`). This
mirrors what `scripts/run_order_supervisor_demo.py` already does
correctly (it calls `evaluate_maker_candidate` per tracked order before
calling `review_order` — the reference-correct pattern already exists in
the repo, one file over from the buggy one, the same pattern as item 3).

**Regression test required**: A resting order whose current (recomputed)
`g_after_if_fill` is below `g_min`, with a regime that has *not* flipped,
must be cancelled via `risk_breach` — currently impossible to trigger
through the ablations harness since `portfolio.G` (not the order's
if-filled G) is what's checked. Separately, an order whose current
recomputed `ev_after` has genuinely fallen below `edge_min` (with
`current_ev_after` no longer hardcoded to 0.0) must be cancelled via
`edge_failure`.

---

## 19. Maker fill chronology (Phase 12 paper executor)

**File**: [execution/simulator.py:115-125](../src/xamarinbot/execution/simulator.py#L115), function `draw_maker_fill()`; used identically in `shadow/runner.py`

**Current behavior**:
```python
def draw_maker_fill(self, order, distance_to_touch_ticks, queue_ahead_shares, horizon_s) -> MakerFillDraw:
    """One reproducible Bernoulli draw for "does this maker order fill
    within horizon_s" ..."""
    rho = fill_probability(...)
    draw = rng.random()
    return MakerFillDraw(filled=draw < rho, ...)
```
This is confirmed to be exactly what the prompt describes: a single
all-or-nothing probabilistic draw evaluated once at TTL expiry (or
immediately, in the no-supervisor dispatch path), not a chronological
walk through actual subsequent book/trade events (queue depletion,
price-level disappearance, partial fills at different times). This is
already self-documented in `docs/PHASE_STATUS.md`'s Phase 7 entry as
*"explicitly an uncalibrated placeholder — no real fill data exists to
estimate it from"* — the prompt's framing adds the specific requirement
(chronological reconstruction from real market events) that makes clear
this must be replaced before any real fill-rate/PnL claim, not just
before live capital.

**Classification**: **INTENTIONAL BUT UNSUITABLE FOR PRODUCTION**
(already correctly labeled as a placeholder in prior documentation; this
audit confirms it is not yet scheduled to change until real market event
data exists to reconstruct fills from, which doesn't exist yet).

**Proposed fix**: Once a real recorder (item 31) exists, rebuild maker
fill simulation as: `SUBMIT -> queue position -> replay real subsequent
trade/book events -> partial fill inference from queue depletion / price-
level disappearance / trades-through-the-level -> more events ->
additional fill / cancel / expiry`, with portfolio state changing only at
the actual simulated fill timestamp. Use the current stochastic draw only
where observable market data is genuinely insufficient, and label any
such inferred fill as uncertain in the journal.

**Regression test required**: Cannot be meaningfully written until real
event data exists to reconstruct fills from; the current stochastic-draw
tests (already in `tests/test_execution.py`) remain valid for what they
test (the probability *model*, not chronological reconstruction) and
should not be deleted.

---

## 20. Phase 12 event-driven cadence

**File**: [shadow/runner.py](../src/xamarinbot/shadow/runner.py), `ShadowRunner.run()`; [events/replay.py:41-56](../src/xamarinbot/events/replay.py#L41), `ReplayClock.decision_points()`

**Current behavior, precisely**: `ShadowRunner` iterates
`clock.decision_points(heartbeat=self.cfg.heartbeat_s)`, which (per
`ReplayClock.decision_points`) is the **union** of every individual
event's own timestamp *and* fixed heartbeat-spaced points — not purely a
10-second-interval loop. In the synthetic dataset (1 event/second per
feed), this already produces an effectively per-second decision cadence
in replay, which is finer than the prompt's framing of "a 10-second
heartbeat" taken alone suggests for *this specific harness*.

**What is genuinely missing**: (a) no differentiation *by event type* —
every decision point does a full feature recompute and full candidate
regeneration regardless of whether the triggering event was a `BOOK_DELTA`
or a `TWAP` update; there is no "material q change" or "risk-state
change" as its own distinct signal. (b) There is no live asyncio/WebSocket
event loop anywhere in the codebase — `ShadowRunner` replays a
pre-recorded, already-complete event log via `clock.decision_points()`,
it does not react to messages arriving in real time on an open socket.
Confirmed by grep: no `asyncio`, `websockets`, or socket-handling code
exists outside the real-adapter *skeletons* in `feeds/polymarket_clob.py`
etc. (item 21), and those skeletons are not wired into `shadow/runner.py`
at all — `ShadowRunner` only ever constructs `MockBookFeed`/mock cursors.

**Classification**: **CONFIRMED DESIGN LIMITATION**, with a correction to
the prompt's framing: the *replay-time* decision cadence is already finer
than "10-second heartbeat" implies (event-timestamp-driven, not
interval-only), but the *live* event-driven loop the prompt is actually
asking about (reacting to real-time WebSocket messages as they arrive,
with per-event-type logic and coalescing/debounce) does not exist at
all — there is no live execution loop of any kind yet, mock or real.

**Proposed fix**: Build a genuine async event loop that subscribes to the
real adapters (once item 21 exists) and dispatches by event type, with a
slower heartbeat retained only as a safety/reconciliation fallback (not
the primary clock). This is new infrastructure, not a modification of
`ShadowRunner`'s current replay-only design (which remains valid for
offline analysis/parity comparison, per Phase 12's original scope).

**Regression test required**: N/A until the live loop exists; forward-looking.

---

## 21-24. Real feeds, market metadata, fee handling, pre-round history

**Item 21 — real feeds**: Already thoroughly self-documented in
`docs/PHASE_STATUS.md`'s "Live adapter confidence" table from Phase 1:
`feeds/polymarket_clob.py` book/market-channel = High confidence
(verified against docs.polymarket.com); market metadata (`get_market_config`)
= Medium, with `_map_tokens_to_sides` explicitly raising
`NotImplementedError` (confirmed present at
[feeds/polymarket_clob.py:111-128](../src/xamarinbot/feeds/polymarket_clob.py#L111));
`feeds/chainlink_twap.py` = Low, endpoint/window not confirmed;
`feeds/polymarket_user.py` = High(WSS)/Unconfirmed(REST). **CONFIRMED**,
matches the prompt's concern exactly, already tracked as an explicit,
named gap rather than a silent one.

**Item 22 — market metadata placeholders**: Same file, same
`NotImplementedError` — confirmed this fails loudly (not silently) per
the module's own stated design, which already satisfies the prompt's
"must fail closed" requirement for the token-mapping gap specifically.
Other fields on `MarketConfig` (`tick_size`, `min_order_size`, `fee_rate`,
`start_ts`/`end_ts`) are typed as required (non-Optional) dataclass
fields fetched from the real endpoint where implemented — not
independently audited line-by-line in this pass beyond confirming the
dataclass shape has no silent defaults. **CONFIRMED** (already
self-documented; remaining work matches "Before real historical data or
live trading" checklist already in `docs/PHASE_STATUS.md`).

**Item 23 — fee handling, checked directly**: `MarketConfig.fee_rate` is
a proper per-market field intended to be fetched at market start (module
docstring: *"Nothing here may be hardcoded by callers"*). However,
`FeeConfig()` (with its `crypto_fee_rate: float = 0.07` fallback) is
constructed **once, at the top of every demo/harness script**
(`scripts/run_shadow_demo.py:61`, `scripts/run_walk_forward_ablation_demo.py:82`,
and by extension every earlier phase's scripts), and **`market_config["fee_rate"]`
is never read to build a per-round `FeeConfig`** anywhere — confirmed by
grep across `walkforward/`, `shadow/`, and every `scripts/run_*.py`. The
field exists on the data model; nothing consumes it. **CONFIRMED DESIGN
LIMITATION** (the architecture supports this; the wiring was never done).

**Item 24 — pre-round history, checked directly**: `features/engine.py`:
```python
spot_w_ago = _value_at_or_before(spot_series, decision_ts - twap_window_s)
s_t_minus_w = spot_w_ago[1] if spot_w_ago is not None else p0  # early-round fallback
```
Confirmed — `p0` is substituted for `S_(t-W)` whenever no observation
exists that far back, which is *always true* near round start unless
pre-round history is separately preserved (it currently is not; the
synthetic generator only emits events from `start_ts` onward, and no
adapter/recorder concept for pre-round BTC/TWAP history exists in the
real-adapter skeletons either). **CONFIRMED DESIGN LIMITATION**, already
partially self-documented (`docs/PHASE_STATUS.md` Known reconstruction
gap referencing the early-round fallback, though not previously framed as
something to *fix* by preserving real history rather than accepting the
approximation).

**Proposed fixes**: (21/22) resolve `_map_tokens_to_sides` and confirm
the Chainlink window/endpoint against current docs before any live
connection, per the existing checklist. (23) read `market_config["fee_rate"]`
at round/market start and construct a per-round `FeeConfig` from it,
keeping `0.07` only as the explicit fallback when metadata is
unavailable and the system isn't failing closed. (24) maintain a
rolling pre-round buffer (BTC/TWAP ticks) sized to the maximum of the
TWAP window, volatility window, and momentum windows, populated before
`start_ts` where real historical ticks are collectible, falling back to
the current `p0` approximation only when genuinely unavailable.

**Regression tests required**: (23) two rounds with different
`market_config["fee_rate"]` values must produce different realized fees
for an identical fill, once wired. (24) a decision point in the first few
seconds of a round, with real pre-round history available, must use that
history for `S_(t-W)` rather than falling back to `p0`.

---

## 25-29. Feature roles, calibration, optimizer framing/objective, marginal optimization

**Item 25 — TWAP/spot distinct roles**: Checked `features/engine.py` —
`G_T`, `G_S`, `L` (lead-lag) are computed as separate, distinct fields on
`FeatureVector` and fed into the model as separate columns (see
`model/features.py`'s `FeatureSet`s — `TWAP_ONLY`, `SPOT_ONLY`,
`COMBINED_LEAD_LAG`, `LEAD_LAG_ONLY` each select different subsets of
these, never collapsing them into a single equal-vote signal).
**NOT A PROBLEM** — matches the prompt's requirement already.

**Item 26 — probability calibration**: Checked `model/` — Platt and
isotonic calibration both implemented (`model/logistic.py` /
`model/calibration.py` per Phase 5), evaluated via Brier score, log loss,
and accuracy on a held-out test split (`reports/`), with
`ModelRegistry` gating promotion on a Brier threshold. Calibration-by-
time-bucket, calibration-by-volatility, and calibration-by-probability-
band reliability curves, plus an explicit uncertainty/`q_safe` concept,
are **not yet implemented** (not found by grep). **PARTIALLY CONFIRMED**:
core calibration machinery exists and is real (not a placeholder); the
requested reliability-curve breakdowns and uncertainty estimation do not
exist yet. **CONFIRMED DESIGN LIMITATION** for the missing
breakdowns/uncertainty specifically; **NOT A PROBLEM** for the core
calibration approach itself, which already does what item 26 asks for at
the top level.

**Item 27 — reframe the optimizer's action space**: Checked
`optimizer/types.py::CandidateAction` — already carries
`purpose, side, mode, price, qty, ttl_s` (missing only an explicit `time`
field, since `decision_ts` is passed as a separate loop variable rather
than a `CandidateAction` field). `purpose` is `OrderPurpose` (`ALPHA`,
`HEDGE` — missing `BUFFER_BUILD`/`REBALANCE`, see item 12-13). `mode` is
`OrderMode` (`FAK, POST_ONLY, WAIT, GTC, GTD, FOK, CANCEL` — a superset of
what item 27 asks, already declared per the module's own docstring
though `GTC/GTD/FOK/CANCEL` aren't yet produced by Phase 8's candidate
generation, only by Phase 9's supervisor). **MOSTLY NOT A PROBLEM** — the
action-space shape the prompt asks for already exists structurally;
item 12-13's `OrderPurpose` gap is the one real piece missing here.

**Item 28 — optimization objective**: Checked `optimizer/controller.py:134`:
`chosen = max(valid, key=lambda c: c.ev_after + self.cfg.lambda_g * c.g_after)`.
This is `E[PnL] + lambda_G*G`, matching *one term* of the requested
`J = E[DeltaPnL] - lambda_tail*TailRisk - lambda_cost*ExecutionCost -
lambda_dd*DrawdownRisk - lambda_churn*OrderChurn`. `churn_penalty` and
`opportunity_cost` (both real, already-wired fields) cover
`lambda_churn*OrderChurn` and roughly `lambda_cost*ExecutionCost`. No
explicit tail-risk (CVaR) or drawdown-risk term exists. This gap is
already self-documented: `docs/PHASE_STATUS.md`'s Phase 11 reconstruction
gap #20 states verbatim *"lambda_g is a single scalar weight on G_after,
not SS18's full J ... with Slippage and other terms left unformalized ...
extending J with the remaining terms is future work once real fill data
exists."* **CONFIRMED DESIGN LIMITATION** (already self-documented, not
newly discovered).

**Item 29 — marginal optimization**: Checked — no function anywhere
computes `DeltaEV` incrementally share-by-share and stops at the
zero-crossing; `taker_quantities`/`maker` sizing are both fixed/discrete
(items 7, 15), not a marginal walk. **CONFIRMED DESIGN LIMITATION**,
same root cause as items 6 and 7 (no marginal-edge machinery exists yet
anywhere in the codebase) rather than a fourth independent gap.

**Classification summary for 25-29**: 25 = NOT A PROBLEM. 26 = split
(core NOT A PROBLEM, breakdowns/uncertainty CONFIRMED DESIGN LIMITATION).
27 = mostly NOT A PROBLEM (one real gap, already covered by item 12-13).
28 = CONFIRMED DESIGN LIMITATION (already self-documented). 29 =
CONFIRMED DESIGN LIMITATION (same root cause as 6/7).

**Proposed fixes**: 26 — add reliability-curve/calibration-by-bucket
reporting and a `q_safe` (e.g. lower-confidence-bound) concept once
real data exists to estimate uncertainty from. 27 — add
`BUFFER_BUILD`/`REBALANCE` to `OrderPurpose` (same fix as item 12-13,
not separate work). 28 — extend the selection objective with tail/
drawdown terms once real fill/PnL data exists to calibrate `lambda_tail`/
`lambda_dd` against (do not guess these, per item 37). 29 — implement
once marginal-edge machinery (item 6/7) exists; this item doesn't need
separate new infrastructure beyond that.

**Regression tests required**: Deferred to items 6, 7, 12-13, and 26 —
this section's tests are the same tests, not additional ones.

---

## 30. MPC scope (GapRegime-only scenario evolution)

**File**: [mpc/scenario.py](../src/xamarinbot/mpc/scenario.py), [mpc/controller.py](../src/xamarinbot/mpc/controller.py)

**Current behavior**: Already thoroughly self-documented in
`docs/PHASE_STATUS.md`'s Phase 10 entry: *"a small discrete scenario tree
that evolves only GapRegime (not the full 54-state RegimeState)"*,
confirmed by reading `mpc/scenario.py::TransitionModel`/`build_transition_model`
directly — transitions are keyed on `GapRegime` alone (6 states), with
`q`, `SpotDirection`, and `CLOBDirection` held fixed across the rollout
horizon rather than evolving. This means the MPC cannot currently
represent or answer the specific timing questions the prompt lists
("wait 300ms? place maker 2 ticks lower? hedge later?") — its horizon
granularity is regime-family transitions, not the fine-grained
book/price/fill-probability evolution those questions require.

**Classification**: **CONFIRMED DESIGN LIMITATION** (already
self-documented as a deliberate Phase 10 scope decision, not a bug;
this audit confirms the prompt's characterization of what it cannot yet
do is accurate).

**Proposed fix**: Per the prompt's own instruction, **do not expand MPC
yet** — first make the one-step/event-driven controller economically and
executionally correct on real data (i.e. items 3-29 above). After that,
rebuild MPC scenarios around distributions of `q_(t+h)`,
`BESTASK_UP/DOWN_(t+h)`, `Spot_(t+h)`, `TWAP_(t+h)`, `CLOB_(t+h)`,
`FillProbability_(t+h)` rather than `GapRegime` transitions alone. This
is explicitly sequenced *last* in the prompt's own implementation order
(item 35 doesn't mention MPC changes at all until real data exists), and
this audit agrees with that sequencing.

**Regression test required**: Deferred — no new MPC work should happen
in this pass per the prompt's own instruction.

---

## 31-38: reporting, promotion criteria, implementation order

Items 31 (real recorder), 32 (re-run ablations on real data), 33
(economic reporting), 34 (promotion criterion), and 35 (implementation
order) are **prescriptive requirements for future work**, not questions
about current repo defects — verified there is currently no real-data
recorder (`recorder`/`Recorder` does not exist as a module; only the
existing `journal/` schema, which records replay/backtest decisions, not
a live non-trading data-collection stream), no real-data ablation rerun
(cannot exist without item 31), and no economic-decomposition report
(`reports/` currently produces per-report metrics, not the requested
`Pair/Buffer Edge + Directional Edge - Fees - Slippage - AdverseSelection`
decomposition). These are correctly scoped as *implementation plan*
items, addressed in the plan section below rather than as audit findings
against existing code.

Item 36 (tests required) and item 37 ("do not optimize synthetic demos")
are process requirements. On item 37 specifically: this audit reviewed
the `lambda_g`/`g_min`/`edge_min` tuning history documented in
`docs/PHASE_STATUS.md` (Phase 11) and confirms it was already framed
there as *"empirically tuned to this dataset's typical magnitudes, not
guessed"* with the reasoning shown — i.e. parameters were adjusted to fix
a **provable scale mismatch** (candidate `g_after` in the hundreds vs
`ev_after` in the tens making the combined score always negative), not
tuned upward/downward merely until synthetic PnL looked better. This is
consistent with item 37's rule but worth stating explicitly since the
prompt calls this out as "critical": no prior tuning in this repo's
history was done by the prohibited pattern ("increase g_min until trades
appear"), though the audit agrees this discipline must continue.

Item 38 (final definition of success) is a statement of intent, not an
audit target — noted and agreed, not applicable to attach file/line
evidence to.

---

# Addendum (reviewer round 2): additional findings A-L

Every item below was independently re-verified against actual source in
this pass, not accepted on the strength of the reviewer's framing alone.
All are **confirmed**, several are more severe or wider-reaching than
originally framed, and one original proposed fix (item 12-13) contained a
genuine mathematical error, now corrected.

## A. Train/evaluation leakage — confirmed, and wider than scoped

**File**: [synthetic/rounds.py:212-221](../src/xamarinbot/synthetic/rounds.py#L212), function `generate_synthetic_dataset()`

**Current behavior, verified directly**:
```python
def generate_synthetic_dataset(store, n_rounds, round_length_s=300.0):
    results = []
    for i in range(n_rounds):
        round_id = f"synthetic-round-{i:04d}"
        bias = [0.0, 7.0, -7.0, 0.0][i % 4]
        result = populate_synthetic_round(
            store, round_id, start_ts=i * (round_length_s + 60.0), round_length_s=round_length_s, bias_bp_per_tick=bias
        )
```
Every call to `generate_synthetic_dataset` starts numbering at `i=0`
regardless of what's already in the target store or in any other store.
`round_id`, `bias`, and `start_ts` are all pure functions of `i` alone —
nothing carries state between calls. `populate_synthetic_round` seeds its
RNG via `seeded_random(round_id, "synthetic-round")` (deterministic on
the string `round_id`). **Therefore two separate calls to
`generate_synthetic_dataset` on two separate `EventStore` objects produce
byte-identical market content for any overlapping index** — "training"
round `synthetic-round-0000` and "evaluation" round `synthetic-round-0000`
are not just same-ID, they are the *same simulated market path*, same
spot ticks, same TWAP, same book, same outcome.

**Confirmed at scale — this is not isolated to one script.** Grepped
every file calling `generate_synthetic_dataset` and checked each
train/eval pair directly:

| File | Train call | Eval call | Overlap |
|---|---|---|---|
| `scripts/run_one_step_controller_demo.py` | `n_rounds=N_TRAIN_ROUNDS` (separate store) | `n_rounds=n_eval_rounds` (separate store) | rounds 0..min(train,eval)-1 fully overlap |
| `scripts/run_order_supervisor_demo.py` | same pattern | same pattern | same |
| `scripts/run_mpc_controller_demo.py` | q-model *and* transition-model each trained on a **separate** `n_rounds=N_TRAIN_ROUNDS` store (harmless redundancy — both are "train") | `n_rounds=n_eval_rounds` (third separate store) | eval overlaps both training stores |
| `scripts/run_walk_forward_ablation_demo.py` | same double-training pattern as MPC demo | `n_rounds=n_rounds` (third store) | eval overlaps both |
| `scripts/run_shadow_demo.py` | `n_rounds=N_TRAIN_ROUNDS=15` | `n_rounds=n_rounds` (default 6) | full overlap (rounds 0-5 identical to training rounds 0-5) |
| `tests/test_walkforward.py` | `trained_model` fixture: `n_rounds=8` | `eval_dataset` fixture: `n_rounds=6` | full overlap (all 6 eval rounds identical to 6 of the 8 training rounds) |
| `tests/test_shadow.py` | `trained_model` fixture: `n_rounds=8` | `eval_dataset` fixture: `n_rounds=3` | full overlap |

**This means every "held-out evaluation" performed anywhere in this
codebase since Phase 8 has actually been evaluating the model on data it
was trained on.** This is strictly worse than the reviewer's framing
suggested — it is not a Phase-11-specific bug, it is the standard pattern
used by nearly every demo script and both of the test suites written in
this and the prior session (`test_walkforward.py`, `test_shadow.py`).

**Mathematical consequence**: any reported metric that compares
train-set behavior to "eval-set" behavior (differ-rates, latency
benchmarks computed on "held-out" rounds, walk-forward window results
where the *rounds themselves* — not just the model — overlap between a
window's train and test segments if the window round_ids happen to
coincide with training round_ids) is measuring memorization, not
generalization, to whatever extent the model has any capacity to
memorize at all.

**Trading consequence**: none of the "n_actions", "differ_rate",
"parity_rate", or PnL numbers reported for Phases 8-12 can be trusted as
evidence of generalization — they are at best pipeline-correctness
checks (which was always the explicit caveat for synthetic data per item
4) and at worst overstate how well the model transfers, because the
"unseen" data literally was seen.

**Classification**: **CONFIRMED BUG**, and the single most
under-scoped item in the original audit — I checked only
`run_walk_forward_ablation_demo.py` at the reviewer's prompting but
should have generalized the check to every script using this pattern
without being asked.

**Proposed fix**: Give `generate_synthetic_dataset` an explicit
`start_index: int = 0` parameter (or accept a `round_ids: list[str]`
directly) so callers can request genuinely disjoint round ranges from a
shared numbering space — e.g. `generate_synthetic_dataset(store, n_rounds=15, start_index=0)`
for train and `generate_synthetic_dataset(store, n_rounds=6, start_index=15)`
for eval. Audit and fix every call site in the table above, not only the
walk-forward demo. This is a mechanical, low-risk fix (one new parameter,
defaulting to today's behavior so nothing breaks silently, plus updating
every call site to pass disjoint ranges) but touches ~9 files.

**Regression test required**: A test asserting
`set(train_round_ids).isdisjoint(set(eval_round_ids))` **and** that the
underlying generated records differ (not just the IDs) — e.g. assert the
first SPOT event's `value` differs between a training round and an
eval round at the same relative index, proving the fix actually changes
the underlying market path, not just the label. Also required per item B
below: assert train/validate/test round IDs are disjoint *within every
walk-forward window*, not only between the top-level train/eval split.

## B. Phase 11 is not a true model walk-forward — confirmed

**File**: [scripts/run_walk_forward_ablation_demo.py:52-57](../scripts/run_walk_forward_ablation_demo.py#L52), [walkforward/sensitivity.py](../src/xamarinbot/walkforward/sensitivity.py)

**Current behavior, verified directly**: `train_q_model()` is called
**once**, before `rolling_windows()` is ever constructed, and the
resulting single `model` object is passed unchanged into every call to
`run_ablation_round`, `sweep_parameter`, and
`parameter_stability_across_windows` for every window. Checked
`sweep_parameter`'s signature (`walkforward/sensitivity.py`) — `model:
LogisticModel | None` is a single parameter, never rebuilt inside the
function; `parameter_stability_across_windows` likewise takes one `model`
and reuses it across every window's `sweep_parameter` call. There is no
per-window `Train_i -> Fit_i -> Validate_i -> Freeze_i -> Test_i` cycle
anywhere — feature standardization, the logistic weights, and (per item C
below) calibration are all fit exactly once, globally, and never touch
window boundaries at all.

**Mathematical consequence**: What Phase 11 currently measures is
"how does *strategy/execution* behavior vary across time-ordered slices
of data, given one fixed model" — a real and useful thing to measure —
**not** "how would a model trained only on each window's own past data
have performed," which is what walk-forward validation is normally
understood to mean and what the roadmap's own Phase 11 spec asks for
("Optimize only on training/validation; lock parameters before each test
segment" — implicitly per-window, not once globally).

**Classification**: **CONFIRMED DESIGN LIMITATION** — a real gap between
what "walk-forward" implies and what's implemented, distinct from (and
compounding) item A's leakage.

**Proposed fix**: Restructure `rolling_windows()` consumers so each
window performs its own `TRAIN -> fit standardization -> fit q-model ->
fit transition model (where used) -> VALIDATE -> calibrate/select
hyperparameters -> FREEZE -> TEST (exactly once, untouched)` cycle, with
none of train/standardization/calibration/transition-probabilities/
strategy-parameters/execution-parameters ever fit or selected using that
window's own test segment. This is a substantial restructuring of
`walkforward/ablations.py` and `sensitivity.py` (they currently assume a
single externally-supplied model), not a small patch.

**Regression test required**: A no-leakage test covering the *entire*
per-window training pipeline (standardization, model fit, calibration,
transition model, hyperparameter selection) — not only
`sweep_parameter()`'s round-id check (already covered by the existing
`test_parameter_stability_across_windows_uses_only_validate_rounds_never_test`
test, which is real but narrower than what's needed here since it only
checks which *rounds* are passed to sweeps, not whether the *model
itself* was fit without seeing test data).

## C. Uncalibrated `q` used throughout Phase 8-12 — confirmed

**File**: every `train_q_model()` function in
`scripts/run_walk_forward_ablation_demo.py:52-57`,
`scripts/run_shadow_demo.py:50-55`,
`scripts/run_mpc_controller_demo.py:50-55`,
`scripts/run_order_supervisor_demo.py:47-52`

**Current behavior, verified directly** (identical pattern in all four
files):
```python
def train_q_model(feature_cfg):
    store = EventStore(":memory:")
    results = generate_synthetic_dataset(store, n_rounds=N_TRAIN_ROUNDS)
    by_fs = build_examples_multi(store, results, feature_cfg, [COMBINED_LEAD_LAG], heartbeat_s=HEARTBEAT_S)
    examples = by_fs[COMBINED_LEAD_LAG.name]
    return fit_logistic_regression([e.x for e in examples], [e.y for e in examples], ...)
```
This returns the **raw** `fit_logistic_regression` output directly — no
calibration step. Confirmed by contrast with Phase 5's own reference
demo, `scripts/run_model_training_demo.py`, which correctly does the
full pipeline (`fit_platt(q_val_raw, y_val)` on a validation split,
applied to test, with the calibrator explicitly chosen over isotonic and
the reasoning documented inline). **Every phase from 8 onward diverges
from Phase 5's own established, correct pattern** and uses raw,
uncalibrated logistic output as `q` in every EV calculation
(`DeltaEV_U(x) = qx - K_U(x)` and everywhere downstream).

**Mathematical consequence**: An uncalibrated `q` can make
`DeltaEV_U(x) = qx - K_U(x)` appear positive when the true (calibrated)
probability would make it negative, or vice versa — every EV-based
selection decision in Phases 8-12's demos and tests is running on a
number that Phase 5's own exit gate ("No production use until calibration
is acceptable") was specifically designed to gate against.

**Classification**: **CONFIRMED BUG** (a validation/design bug, per the
reviewer's requested classification) — this is not a "missing nice-to-have,"
it's skipping a gate Phase 5 itself already implemented and enforces via
`ModelRegistry`, just not through these particular call sites.

**Proposed fix**: Replace every `train_q_model()` above with the same
train -> validate -> calibrate (Platt, per Phase 5's documented reasoning
for this dataset) -> freeze pattern already correctly implemented in
`run_model_training_demo.py`, ideally by extracting that pattern into a
shared helper both Phase 5's demo and Phases 8-12's demos/harnesses call,
rather than four more copies of a train/calibrate pipeline. Phase 12B's
real-data walk-forward (item 32) must evaluate the exact calibrated model
object the controller actually uses, not a separately-fit raw one.

**Regression test required**: Assert the `model` object used inside
`run_ablation_round`/`ShadowRunner` has `calibrator is not None` (or
equivalent), and that `predict_proba` output differs from the raw
logistic sigmoid output on at least one held-out example (proving
calibration is actually applied, not just constructed and discarded).

## D. Second baseline harness bug: absolute vs. elapsed time — confirmed, compounds with item 3

**File**: [walkforward/ablations.py:153](../src/xamarinbot/walkforward/ablations.py#L153), function `_run_baseline_round()`

**Current behavior, verified directly**:
```python
inputs = BaselineInputs(
    t=decision_ts, p0=p0, twap=twap_obs.value, clob_mid=mid, clob_mid_prev=mid_prev,
    ...
)
```
`t=decision_ts` passes the **absolute** replay timestamp.
`baseline/strategy.py::decide()` checks
`cfg.decision_window_start_s <= inputs.t <= cfg.decision_window_end_s`
(default `[15, 270]`), which is only meaningful for **elapsed round
time**, not absolute time. Confirmed against `synthetic/rounds.py`'s own
round layout (`start_ts=i * (round_length_s + 60.0)`, i.e. round 0 spans
`[0, 300]`, round 1 spans `[360, 660]`, round 2 spans `[720, 1020]`, ...).
**Every round except round index 0 has `decision_ts` values entirely
outside `[15, 270]` for its whole duration**, so
`OUTSIDE_DECISION_WINDOW` fires at every single decision point for every
round after the first, regardless of whether item 3's `spot_prev` bug is
fixed.

**Confirmed as isolated to the same Phase 11 harness function, same
pattern as item 3**: Phase 0's original `scripts/run_baseline_replay.py:150`
already does this correctly —
`t=decision_time - market_config.start_ts` — a third instance (after
`spot_prev` and this) of `_run_baseline_round` diverging from an
already-correct reference implementation one file over.

**Mathematical consequence**: combined with item 3, the baseline placed
**zero possible trades in ~(N-1)/N of all evaluated rounds** purely from
this bug, independent of and in addition to the unanimity-breaking
`spot_prev` bug. Fixing `spot_prev` alone would not have produced a
working baseline in any Phase 11 ablation matrix that used more than one
round — round 0 would show whatever the spot_prev fix newly enables,
every other round would still show zero from this bug alone.

**Classification**: **CONFIRMED BUG**, second high-severity Phase 11
baseline harness defect, must be fixed together with item 3 (fixing one
without the other still leaves the baseline arm non-functional for
`n_rounds > 1`).

**Proposed fix**: `t=decision_ts - market_config["start_ts"]`, reading
`start_ts` from the same `market_config` dict already loaded via
`next(e.payload for e in events if e.event_type is EventType.MARKET_CONFIG)`
elsewhere in this module (no new data dependency needed).

**Regression test required**: Three rounds with identical relative
0-300s market paths but round-start timestamps of `0`, `360`, and a large
realistic epoch value (e.g. `1_700_000_000.0`) must produce **identical**
baseline decisions/skip-reasons at every relative timestamp — currently
would produce a working baseline only for the first.

## E. Execution-path inconsistency across Phase 11/12 — confirmed, wider root cause identified

**File**: [walkforward/ablations.py:235-239](../src/xamarinbot/walkforward/ablations.py#L235), [shadow/runner.py:165-168](../src/xamarinbot/shadow/runner.py#L165), [walkforward/ablations.py:158-161](../src/xamarinbot/walkforward/ablations.py#L158) (baseline)

**V2/optimizer arms, current behavior, verified directly** (identical in
both `ablations.py` and `shadow/runner.py`):
```python
if chosen.mode is OrderMode.FAK and chosen.expected_fill > 0:
    fee = fee_config.taker_fee(chosen.expected_fill, chosen.price)
    portfolio = apply_fill(portfolio, Fill(chosen.side, chosen.price, chosen.expected_fill, LiquidityRole.TAKER, fee))
```
`chosen.price`/`chosen.expected_fill` *do* come from a real depth-walk
(`evaluate_taker_candidate` calls `execution/taker.py::walk_depth`
against the actual book at `decision_ts` before this dispatch code ever
runs), so partial-fill sizing and price-impact realism **are** present.
What's missing is Phase 7's `OrderState` lifecycle
(`submit_taker_order -> PENDING_DELAY -> resolve_pending` at
`matched_ts`) — the fill is applied synchronously at `decision_ts`
with no delay and no book revalidation, bypassing the exact machinery
`execution/simulator.py::submit_taker_order`/`resolve_pending` exists
to provide and that `tests/test_execution.py` already covers correctly
in isolation.

**Important scoping correction to the reviewer's framing**: this
shortcut is **not new to Phase 11/12** — grepped and confirmed
`scripts/run_one_step_controller_demo.py` (Phase 8's own reference demo)
uses the identical direct-apply pattern
(`if chosen.mode is OrderMode.FAK:` -> apply fill directly), never
calling `submit_taker_order`. **Every phase from 8 onward has bypassed
the delay/revalidation lifecycle**, not just Phase 11/12. Also confirmed:
`ExecutionConfig.taker_delay_ms` defaults to `0.0` everywhere it's
constructed (its own docstring says *"read from MarketConfig.taker_delay_ms
per round in practice"* — but per item 23, nothing actually does that
read anywhere in the codebase). At `taker_delay_ms=0.0`, submit-then-
immediately-resolve is mathematically equivalent to direct-apply, so
**this has not silently corrupted any PnL number to date** — but the
moment a real market's actual delay (Strategy doc: 250ms on crypto
markets) is wired in via item 23's fix, this shortcut would **silently
ignore it entirely**, since the direct-apply path never reads
`taker_delay_ms` at all.

**Baseline, current behavior, verified directly**
(`walkforward/ablations.py:158-161`):
```python
result = baseline_decide(inputs, portfolio.U, portfolio.D, portfolio.C, cfg)
if result.order is not None:
    fee = fee_config.fee_for(result.order.role, result.order.quantity, result.order.price)
    portfolio = apply_fill(portfolio, Fill(result.order.side, result.order.price, result.order.quantity, result.order.role, fee))
```
`result.order.price` is the pre-computed marketable-limit price
(`min(1.0, best_ask + limit_delta)`) and `result.order.quantity` is the
*requested* quantity — confirmed, **no depth-walk at all**, always a
full fill at the limit price. This is a materially easier execution
assumption than the V2 arms (which at least depth-walk), confirmed
exactly as the reviewer describes, and independently makes any
baseline-vs-V2 PnL comparison invalid on execution-realism grounds
*even after* items 3 and D are fixed.

**Classification**: **CONFIRMED BUG** for the baseline's full-fill
assumption (a real, wrong simplification, not merely a "known
placeholder" — it was never labeled as such anywhere). **CONFIRMED
DESIGN LIMITATION** for the V2 arms' delay/revalidation bypass (real
depth-walk cost realism is present; only the timing dimension is
skipped, and it has been inert-by-coincidence rather than actively wrong
so far because of the `taker_delay_ms=0.0` default everywhere).

**Proposed fix**: Route both the baseline and every V2 arm's taker fills
through one common execution path — `submit_taker_order` ->
`resolve_pending` at the correct `matched_ts`, using
`walk_depth`-derived partial-fill sizing for *both* arms (the baseline
currently gets none at all). This directly serves item L (common
execution layer across ablation arms) — the fix for E and the fix for L
are the same piece of work, not two separate ones.

**Regression test required**: A round with `taker_delay_ms > 0` and a
book that moves between `decision_ts` and `decision_ts + taker_delay_ms`
must produce a different (revalidated) fill than a naive direct-apply —
proving the delay actually matters once wired in, and regression-proofing
against silently reverting to the shortcut. Separately, a test that the
baseline's fill quantity is bounded by actual depth at its execution
price, not always the full requested quantity.

## F-H. Mathematical correction: `BUFFER_BUILD` / one-sided vs. two-sided `ΔG`

**This is a correction to this audit's own item 12-13 proposed fix, not
a new code defect** — no code implementing `BUFFER_BUILD` exists yet
(confirmed: `OrderPurpose` still only has `ALPHA`/`HEDGE`), so there is
nothing in `main` to classify as buggy here. This section verifies the
reviewer's corrected math is right and adopts it, superseding the
original proposal.

**The error in the original proposal**: item 12-13's "Proposed fix" used
`DeltaG_pair(x) = x - K_U(x) - K_D(x)` to describe acquiring `x` shares
on a *single* (the cheaper/under-represented) side. That formula is only
valid for a **simultaneous two-sided** acquisition.

**Proof, worked from the kernel definition `G = min(U,D) - C`**:

*Two-sided case* (`U'=U+x, D'=D+x, C'=C+K_U(x)+K_D(x)`):
```
ΔG_both(x) = min(U+x, D+x) - (C+K_U(x)+K_D(x)) - [min(U,D) - C]
           = [min(U,D) + x] - min(U,D) - K_U(x) - K_D(x)      (since min(a+x,b+x) = min(a,b)+x)
           = x - K_U(x) - K_D(x)
```
This confirms the original formula is correct — **but only for this
case**, which is not what "acquire x on the cheaper side" (a one-sided
fill) describes.

*One-sided UP case* (`U'=U+x, D'=D, C'=C+K_U(x)`):
```
ΔG_U(x) = min(U+x, D) - (C+K_U(x)) - [min(U,D) - C]
        = min(U+x, D) - min(U,D) - K_U(x)
```
This is the reviewer's boxed formula, confirmed correct by direct
substitution into the kernel — not a new/different kernel, the exact
same `G = min(U,D) - C` the Phase 3 property tests already cover, just
evaluated for a one-sided delta instead of assumed-equal-both-sides.

*One-sided DOWN case*, symmetric:
`ΔG_D(x) = min(U, D+x) - min(U,D) - K_D(x)`.

*Special case, verified*: if `U < D` and `x <= D-U`, then
`U+x <= D`, so `min(U+x,D) = U+x`, and `min(U,D) = U`. Substituting:
`ΔG_U(x) = (U+x) - U - K_U(x) = x - K_U(x)` — confirmed, matches the
reviewer's boxed special case exactly. The economic reading is correct
too: existing DOWN inventory already priced into `min(U,D)` doesn't need
to be "re-bought" — only the *new* UP shares' cost is charged against the
buffer improvement while `U` stays the binding side of the min.

**Classification**: **CONFIRMED** — the reviewer's correction is
mathematically right, proven directly against the existing, tested `G`
kernel (no change to Phase 3's math itself, only to how a proposed new
candidate type would compute its own `ΔG`).

**Item G, checked separately**: `ΔEV_U(x) = q*x - K_U(x)` and
`ΔG_U(x)` (above) are independent quantities — confirmed there's no
algebraic identity forcing them to share a sign (e.g. `ΔG_U(x) > 0`
requires `min(U+x,D)-min(U,D) > K_U(x)`, a pure portfolio-geometry
condition, while `ΔEV_U(x) > 0` requires `q*x > K_U(x)`, a pure
probability/cost condition — nothing ties `q` to `min(U+x,D)-min(U,D)`).
**Confirmed correct** — a `BUFFER_BUILD` candidate with `ΔG>0, ΔEV<0` is
a real, non-contradictory possibility and must be evaluated/labeled as
such, not conflated with `ALPHA`'s profitability meaning.

**Item H, checked against current code**: confirmed no atomic
matched-pair execution exists or is proposed anywhere (`generate_hedge_candidate`
submits one single-sided order; nothing in the codebase submits two
orders as a unit). The reviewer's leg-risk concern is therefore
correctly forward-looking, not a claim about existing code. Agreed:
`BUFFER_BUILD`'s two-sided variant (if implemented at all — the one-sided
`ΔG_U`/`ΔG_D` case above is likely sufficient for the "accumulate cheap
opposite-side inventory over time" strategy the prompt actually
describes, since it doesn't require simultaneity) must model
`FirstLeg -> TemporaryPortfolio -> SecondLeg` explicitly if a genuinely
simultaneous two-sided candidate is ever built, rather than assuming
atomicity.

**Proposed fix (supersedes item 12-13's original)**: Implement
`ΔG_U(x)`/`ΔG_D(x)` (one-sided) as the primary `BUFFER_BUILD` formula —
this directly matches how the strategy actually accumulates inventory
(sequential single-sided fills over the round, not simultaneous pairs).
Keep `ΔG_both(x)` available only if/when a genuinely simultaneous
two-sided order type is built, modeled with explicit leg-risk per item H
rather than assumed atomic. Compute `ΔEV_U(x)`/`ΔEV_D(x)` independently
per item G, and surface both `delta_ev` and `delta_g` as separate fields
on any `BUFFER_BUILD`/`HEDGE` candidate so the optimizer (and any human
reviewing the journal) can see risk-improvement and expected-value
contribution separately, never collapsed into one "profitable" label.

**Regression tests required**: `ΔG_U(x) == min(U+x,D) - min(U,D) - K_U(x)`
and the symmetric DOWN case, each verified directly against
`PortfolioState.G` computed before/after a real simulated fill (not
re-derived independently — i.e. the test must call `apply_fill` and
compare `after.G - before.G` to the closed-form formula, the same
pattern `tests/test_portfolio_math.py` already uses for the existing
identities). The special-case formula (`ΔG_U(x) = x - K_U(x)` when
`x <= D-U`) as its own explicit test case, not just covered incidentally
by the general formula's property test.

## I. Do not activate `p_min` while fixing the favored-side bug — adopted

No code contradicts this — item 10's proposed fix already only changes
*how* `favored_side` is computed, not whether `p_min` becomes active;
`OneStepConfig.p_min` stays `None` by default regardless. **Adopted as
an explicit constraint on the implementation plan**: fixing
`_favored_side()` must not be paired with any change that starts passing
a non-`None` `p_min` into any demo/ablation config. Re-confirmed via the
same grep as item 10 — `p_min` is set nowhere today; the fix plan will
keep it that way unless a specific risk-policy reason is given separately.

## J. Do not invent a sophisticated dynamic `g_min` on synthetic data — adopted

Consistent with and reinforces item 11's own original conclusion ("Do not
invent the final production formula yet... calibrate it using validation
data only... No arbitrary g_min = -100 should be interpreted as
inherently optimal"). **Adopted, no change needed to item 11's
conclusion** — the correctness-tranche work is limited to exposing a
clean, explicit, bankroll-relative-*capable* config shape that fails
closed when required inputs are absent, not to fitting or guessing the
final formula's weights.

## K. Revised sequencing — adopted, see corrected plan below

Superseded the original "Exact Phase 12B implementation plan" — see the
rebuilt Tranche 1-5 structure in the Summary section below, which follows
this section's ordering exactly, with items 3/D, A, B, C, 9, 5, 6, 7, 8,
10, 18, and E/L now explicitly grouped into Tranche 1 ("correctness /
invalid-result repair") before any of Tranche 2's portfolio-control
architecture work begins.

## L. One common execution layer across all Phase 11 ablation arms — adopted, same fix as item E

Confirmed via item E's investigation: baseline currently has *no* depth
realism (full fill at limit price) while V2 arms have depth-walk realism
but no delay/revalidation realism — neither arm's execution model matches
the other today, so any `PnL_A - PnL_B` comparison between them
(including the already-published Phase 11 ablation matrix numbers, which
compare `1_baseline_unanimous` against arms 2-8) cannot be attributed to
the strategy difference alone. **This is not a new finding distinct from
E — it is E's baseline-vs-V2 half, restated as a cross-arm requirement.**
The fix is identical: route every arm's taker fills through the same
`submit_taker_order`/`resolve_pending`/`walk_depth` path.

---

# Summary (per the prompt's required closing sections)

## Which issues are confirmed

**CONFIRMED BUG** (real defects, not design tradeoffs):
- Item 3 — baseline `spot_prev` bug in `walkforward/ablations.py`, invalidates every Phase 11 baseline comparison to date.
- Item 7 — taker candidate sizing has no intermediate/marginal/risk-budget-derived quantities, only raw depth-level cumulative sums; already reproduced empirical consequence on record (ablation #6).
- Item 9 — `spend_cap` checked per-order, not cumulatively per round.
- Item 10 — favored-side inference uses payoff geometry (`Pi_U >= Pi_D`), not prediction; currently dormant (`p_min` never configured) but would misfire immediately if enabled.
- Item 18 — Phase 11 ablations 7/8 call the supervisor with a hardcoded `current_ev_after=0.0` and the wrong (`unconditional` vs `if-filled`) `G` value; `current_optimal_ev` never passed at all, so `REPLACE` can never fire in that harness.
- **Addendum A — train/evaluation leakage**: every train/eval split in the
  codebase (9 files: 6 demo scripts, 2 test files, plus the walk-forward
  harness) generates "held-out" data from the same deterministic
  round-numbering scheme used for training, producing byte-identical
  market content for overlapping indices. The single most under-scoped
  finding in the original audit pass.
- **Addendum C — uncalibrated `q`**: every `train_q_model()` in Phases
  8-12's demos/harnesses returns raw `fit_logistic_regression` output,
  skipping the calibration step Phase 5's own reference demo already
  implements correctly and that Phase 5's own exit gate requires.
- **Addendum D — baseline absolute-vs-elapsed time**: `_run_baseline_round`
  passes absolute `decision_ts` where elapsed round time is required,
  making the baseline fail its decision-window gate for every round
  except the one starting at `t=0`. Compounds with item 3 — both must be
  fixed together for the baseline arm to function at all across more than
  one round.
- **Addendum E — baseline execution realism**: the baseline's fill
  application assumes a full fill at the limit price with no depth-walk
  at all, a materially easier execution assumption than every V2 arm —
  confirmed, independent of items 3/D, and invalidates baseline-vs-V2 PnL
  comparisons on execution-realism grounds alone.

**CONFIRMED DESIGN LIMITATION** (real, verified gaps, several already
partially self-documented and now given precise root causes) additionally
includes, from the addendum:
- **Addendum B** — no per-window model retraining; Phase 11 currently
  measures strategy/execution sensitivity around one fixed, globally-fit
  model, not true walk-forward model validation.
- **Addendum E (V2-arm half)** — taker fills in every Phase 8-12
  controller path bypass Phase 7's `OrderState` delay/revalidation
  lifecycle, inert-by-coincidence today (`taker_delay_ms=0.0` everywhere)
  but silently ignores real delay the moment item 23 wires in per-market
  config.

**CONFIRMED DESIGN LIMITATION** (real, verified gaps vs. the intended
architecture, several already partially self-documented in
`docs/PHASE_STATUS.md` and now given precise root causes):
Items 4, 5, 6, 8 (translation layer half), 11, 12-13 (the largest single
gap found), 14, 15, 16 (already self-documented, unresolved), 17, 20
(with the correction noted), 21-24, 26 (breakdowns/uncertainty half), 28
(already self-documented), 29.

**INTENTIONAL BUT UNSUITABLE FOR PRODUCTION**:
Item 8 (worst-price question specifically — correct for a static-snapshot
backtest, must change before live submission), item 19 (maker fill
chronology — already labeled a placeholder, correctly so).

**NOT A PROBLEM**:
Item 0 (no fixed-dollar logic exists to remove), item 1 (portfolio kernel
correct), item 25 (TWAP/spot roles already distinct), most of item 27
(action-space shape already close to what's requested).

**NEEDS REAL DATA TO DETERMINE**:
Any claim about actual profitability, optimal parameter values, or
whether MPC/maker/portfolio-repair "work" economically — consistent with
item 4's finding that the synthetic generator cannot support such claims
regardless of how correct the surrounding code becomes.

## Which claims I dispute and why

- **Item 20's framing** ("a 10-second heartbeat... is not an appropriate
  primary decision clock") is correct in spirit but imprecise about this
  specific repo: `ReplayClock.decision_points()` already merges every
  individual event's own timestamp into the schedule, not just fixed
  10-second points, so the *replay-time* cadence is already much finer
  than "10-second heartbeat" alone implies. The real gap — no live
  async event loop exists at all — is confirmed and I agree that's the
  substantive issue; I'm only disputing the specific characterization of
  the current replay cadence, not the conclusion.
- **Item 16's characterization** as needing correction slightly
  understates that the *EV* term is already correctly `rho`-weighted
  today; the actual defect is narrower and more specific (the `g_after`
  term used in the same selection score is not correspondingly weighted)
  — already self-documented in `docs/PHASE_STATUS.md`, and I want to be
  precise that this isn't "maker utility is unweighted," it's "one of two
  terms in the same score has inconsistent weighting."
- No other claim in the original 38-item prompt was found to be factually
  incorrect against the current code — every other numbered item's core
  technical claim was verified true by direct inspection, not merely
  plausible.
- Of the reviewer's round-2 findings (A-L), none of A-E or I-L were found
  factually incorrect — every one was independently re-verified against
  actual code (not accepted on the strength of the reviewer's framing
  alone) and confirmed true, several turning out **more severe or
  wider-reaching** than the reviewer's own framing (Addendum A's leakage
  is repo-wide across 9 files, not one script; Addendum E's baseline
  execution gap is independent of and additional to the depth-walk
  question the reviewer raised for the V2 arms). The one genuine error
  found in this round was **in this audit's own** original item 12-13
  proposed fix (the reviewer's item F correctly identified it) — see
  Addendum F-H for the correction, now adopted.

## Additional flaws discovered during this audit (not named in the prompt)

- `max_directional_spend()` (`portfolio/math.py`) has existed since
  Phase 3, is unit-tested in isolation, but is **never called** from
  anywhere in the candidate-generation path — dead code that item 7's fix
  should actually wire in, rather than reimplementing equivalent logic.
- `RegimeConfig`/`SupervisorConfig` defaults are, like `OneStepConfig.g_min`,
  hardcoded magic constants tuned to this specific synthetic dataset
  (e.g. `SupervisorConfig.g_min` default `-1_000_000.0` — a sentinel
  effectively meaning "never trigger risk_breach unless explicitly
  configured," which is itself worth flagging: a caller who forgets to
  pass a real `g_min` to the supervisor silently gets no risk-breach
  protection at all, rather than a fail-closed error).
- The favored-side bug (item 10) and the supervisor-placeholder bug
  (item 18) share a common shape: both are currently **dormant** because
  the features that would expose them (`p_min`, dynamic re-evaluation)
  are unused/unconfigured in every existing demo and test. This is worth
  flagging as a pattern: several parts of this codebase are only as
  correct as their test coverage, and coverage has followed "what the
  demos exercise," not "what the interface allows a caller to configure."

## Exact Phase 12B implementation plan (proposed — not yet started)

**Superseded by the reviewer's Tranche structure (Addendum K), adopted
in full.** The original single-sequence plan (previous revision of this
document) is replaced below — correctness/invalid-result repair is now
explicitly front-loaded into its own tranche, before any portfolio-control
architecture work (BUFFER_BUILD etc.) begins, and real-market-foundation
work is separated from real-data calibration, which is separated from
economics reporting. Every step keeps existing passing tests green (item
36's "no existing test may be deleted simply because the architecture
changes" rule); semantics-changing steps get their old test updated in
place with an inline comment explaining why, alongside new tests, not a
silent deletion.

### Tranche 1 — Correctness / invalid-result repair

1. ~~Amend `docs/PHASE_12B_AUDIT.md` with Addendum A-L~~ — this revision.
2. Fix baseline `spot_prev` (item 3).
3. Fix baseline elapsed-time `t` (Addendum D) — **must land together
   with step 2**; fixing either alone still leaves the baseline arm
   non-functional (item 3 alone: still zero trades outside round 0;
   Addendum D alone: unanimity still impossible).
4. Eliminate train/evaluation round overlap (Addendum A) — add
   `start_index`/`round_ids` support to `generate_synthetic_dataset`,
   fix all 9 identified call sites (6 demo scripts, 2 test files, the
   walk-forward harness itself).
5. Rebuild real walk-forward model fitting/calibration boundaries
   (Addendum B) — per-window `TRAIN -> fit -> VALIDATE -> calibrate ->
   FREEZE -> TEST`, folding in calibration (Addendum C) so the
   per-window `Fit_i` step includes Platt calibration on that window's
   own validation segment, not a separately-fit raw model.
6. Fix cumulative round `spend_cap` (item 9).
7. Rename/clarify `ev_after` → `delta_ev` (item 5).
8. Add marginal edge, separate from total EV (item 6).
9. Implement partial/risk/depth-aware taker sizing (item 7), wiring in
   the currently-dead `max_directional_spend`.
10. Derive real worst-price protection (item 8, backtest-relevant half);
    defer the dollar-amount translation layer to the real-adapter tranche.
11. Fix favored-side semantics (item 10) **without** activating `p_min`
    (Addendum I) — `OneStepConfig.p_min` stays `None` in every existing
    config; the fix changes only how `_favored_side` would compute a
    value if `p_min` were ever set.
12. Fix Phase 11 supervisor current-order reevaluation (item 18) —
    mirror the already-correct pattern in
    `scripts/run_order_supervisor_demo.py`.
13. Route all Phase 11/12 taker fills through one common, chronological
    execution path (Addendum E + L combined — same fix serves both):
    `submit_taker_order -> resolve_pending` with real `walk_depth`
    sizing, applied identically to the baseline arm (which currently has
    none) and every V2 arm (which currently has depth-walk sizing but no
    delay/revalidation).

**After this tranche**: rerun all tests and the synthetic suite **only
as correctness tests**. Do not report synthetic profitability, per items
4/37 and Addendum A's confirmation that no "held-out" evaluation to date
has actually been held out.

### Tranche 2 — Exact portfolio-control architecture

14. Implement `ΔG_U(x)`/`ΔG_D(x)` (one-sided, per Addendum F-H's
    correction — **not** the two-sided `ΔG_pair` formula this audit
    originally and incorrectly proposed for a one-sided fill), with
    `ΔEV_U(x)`/`ΔEV_D(x)` computed independently per Addendum G so
    `ΔG>0, ΔEV<0` candidates are representable and correctly labeled, not
    conflated with `ALPHA`'s profitability meaning.
15. `BUFFER_BUILD`/`HEDGE`/`REBALANCE` purpose separation (items 12-13),
    built on step 14's corrected formulas, generated independent of a
    `G < g_min` breach.
16. Sequential matched-pair / leg-risk state (Addendum H) — model
    `FirstLeg -> TemporaryPortfolio -> SecondLeg` explicitly if/when a
    genuinely simultaneous two-sided candidate is built; the one-sided
    `BUFFER_BUILD` from step 14-15 does not require this since it isn't
    atomic-pair-dependent.
17. Soft-regime mode alongside the current hard-gate, kept as an
    explicit ablation flag (item 14).
18. Maker hold/cancel/replace economic reevaluation (item 17,
    `V_hold`/`V_cancel`/`V_replace`).
19. Cancellation hysteresis/persistence (item 17, continued).
20. Correct maker probability/risk utility weighting consistency
    (item 16).
21. Dynamic maker price/quantity/TTL (item 15).
22. Chronological maker partial fills (item 19) — the *design*, not
    yet backed by real event data; blocked on Tranche 3's recorder for
    real fill reconstruction, but the chronological state-machine
    structure (submit -> queue -> partial -> more events -> resolve) can
    be built now and tested against hand-constructed event sequences,
    the same style `tests/test_execution.py` already uses.

Do not tune any of steps 14-22 to synthetic PnL (item 37 / Addendum J) —
these are architecture/correctness changes, verified by property tests
against the exact kernel (Addendum F-H's proof method) and hand-
constructed scenarios, not by watching synthetic PnL move.

### Tranche 3 — Real market foundation

23. Finish actual market metadata adapter (`_map_tokens_to_sides`, item 22).
24. Finish Chainlink TWAP mapping/authentication (item 21).
25. Wire per-market fee config (item 23) and pre-round BTC/TWAP history
    (item 24).
26. Implement low-latency BTC event stream + Polymarket CLOB/user/order
    WebSockets (item 21), and the dollar-amount translation layer
    deferred from Tranche 1 step 10.
27. Build a non-trading real round recorder (item 31); validate source
    vs. receive timestamps (already-correct `event_time`/`recv_ts`
    machinery from Phases 2/12 — this step is about *recording* real
    data through it, not changing the causal model).
28. Collect real BTC five-minute rounds continuously.

### Tranche 4 — Real-data calibration

Only after enough real data exists from Tranche 3:

29. Fit and calibrate `q` using true per-window walk-forward (Tranche
    1 step 5's rebuilt pipeline, now against real rounds).
30. Estimate `q` uncertainty (`q_safe`, item 26).
31. Estimate maker fill hazard / adverse selection from real fills
    (replacing item 19's stochastic-draw placeholder with the
    chronological reconstruction from Tranche 2 step 22, now backed by
    real events).
32. Estimate latency/slippage distributions.
33. Calibrate regime persistence (informing Tranche 2's hysteresis
    thresholds with real data instead of a guessed constant).
34. Calibrate risk parameters — **only now**, per item 11/Addendum J,
    build the bankroll-relative `g_min` formula, calibrated against real
    validation data.
35. Compare hard regime vs. soft regime (Tranche 2 step 17's ablation)
    on real data.
36. Compare static vs. dynamic quantity/TTL/execution choices on real
    data.
37. Re-run the full ablation matrix (item 32) on real recorded rounds,
    including the new ablations item 32 names (hard vs. soft regime
    gating, proactive buffer on/off, fixed vs. optimized quantity, fixed
    vs. dynamic TTL, market vs. mixed maker/taker, static vs.
    bankroll-relative risk floor) — using the one common execution layer
    from Tranche 1 step 13 so every arm differs only in the strategy
    dimension being ablated (item L).

### Tranche 5 — Real shadow economics

38. Build the genuine event-driven live loop (item 20) — real async
    dispatch by event type, heartbeat retained only as a
    safety/reconciliation fallback.
39. Run true event-driven shadow trading against the real feeds from
    Tranche 3.
40. Produce the Phase 12B economic report (item 33) with the requested
    net PnL / PnL-per-round / trade frequency / profit factor / average
    winner-loser / largest loss / max drawdown / CVaR / fees / slippage
    / maker fill rate / adverse selection / partial fills / cancel
    regret / G distribution / R distribution / Pi_U/Pi_D distribution /
    buffer contribution / directional contribution decomposition
    (`Pair/Buffer Edge + Directional Edge - Fees - Slippage - AdverseSelection`).

**Only after Tranche 5 may Phase 13 be proposed**, per item 34's
promotion criterion — no arbitrary required dollar PnL per round, per
item 0/38.

## Files that will change

**Tranche 1**: `synthetic/rounds.py` (leakage fix — new parameter,
default-compatible), `walkforward/ablations.py` (baseline `spot_prev` +
elapsed-time fixes, common execution path, per-window model fitting),
`walkforward/sensitivity.py` (per-window fitting), `optimizer/candidates.py`,
`optimizer/controller.py`, `optimizer/config.py`, `optimizer/types.py`,
`portfolio/math.py`, `supervisor/supervisor.py`, `execution/simulator.py`
(wiring `submit_taker_order` into the ablations/shadow dispatch paths),
`shadow/runner.py`, plus every demo script identified in Addendum A's
table (`scripts/run_one_step_controller_demo.py`,
`scripts/run_order_supervisor_demo.py`, `scripts/run_mpc_controller_demo.py`,
`scripts/run_walk_forward_ablation_demo.py`, `scripts/run_shadow_demo.py`)
and both affected test files (`tests/test_walkforward.py`,
`tests/test_shadow.py`), and a new shared calibration helper (extracted
from `scripts/run_model_training_demo.py`'s existing correct pattern) —
this tranche alone is wider-reaching than the original plan's Tranche 1
equivalent because of Addendum A's 9-file scope.

**Tranche 2**: `optimizer/candidates.py`, `optimizer/types.py`
(`OrderPurpose` additions), `portfolio/math.py` (new `ΔG_U`/`ΔG_D`
functions alongside, not replacing, the existing kernel),
`supervisor/supervisor.py`, `supervisor/predicates.py`,
`supervisor/config.py`, `regime/matrix.py` (soft-prior mode, additive),
`optimizer/config.py` (dynamic maker params).

**Tranche 3**: `feeds/polymarket_clob.py`, `feeds/chainlink_twap.py`,
`feeds/polymarket_user.py`, `feeds/spot_composite.py`, a new `recorder/`
module.

**Tranche 4-5**: `model/` (per-window calibration pipeline), a new
`reports/economic_report.py`, a new live event-loop module (name TBD,
likely `live/runner.py` alongside the existing `shadow/`).

Plus the corresponding test files for every item above and
`docs/PHASE_STATUS.md`.

## Mathematical invariants that will remain untouched

- `Pi_U = U - C`, `Pi_D = D - C`, `G = min(Pi_U, Pi_D)`, `R = Pi_U - Pi_D`
  (item 1 — confirmed correct, not touched).
- The `EV_after_total - EV_before_total == q*ΔU + (1-q)*ΔD - ΔC` identity
  (item 5 — the *value* stays; only the *name* changes to `delta_ev`, and
  a test will pin the identity down explicitly for the first time).
- `ΔG_U(x) = min(U+x,D) - min(U,D) - K_U(x)` and the symmetric DOWN case
  (Addendum F-H — new code, but the identity is exact and proven directly
  against the existing `G = min(U,D) - C` kernel above, not an
  approximation; **this replaces the previous revision's incorrect
  `ΔG_pair(x) = x - K_U(x) - K_D(x)` for the one-sided case** — that
  formula is kept, correctly scoped to a genuinely simultaneous two-sided
  fill only, per Addendum H).
- Causal replay: `event_time <= decision_time` (Phase 2) and
  `recv_ts <= decision_time` (Phase 12's stricter live gate) both stay
  exactly as implemented — no item in this audit or its addendum calls
  either into question.
- Every existing Hypothesis property test in `tests/test_portfolio_math.py`,
  `tests/test_optimizer.py`, and `tests/test_execution.py` continues to
  hold under the proposed changes (none of the proposed fixes alter the
  fill-simulation or constraint-evaluation math itself, only what
  candidates are generated, what model produced `q`, what data a model
  was fit/evaluated on, and what values are compared against constraints).

---

# Tranche 1 — Implemented (round 3)

**Approved and implemented.** Every Tranche 1 item (the reviewer's 16-item
approval list) is done, tested, and verified via the full test suite plus
every affected demo script run end-to-end. Full detail — exact files
changed, each bug and its exact fix, the formulas implemented, the
train/validate/test and execution architecture before vs. after, every
new regression test, the complete suite result, and remaining known
limitations — was delivered as the chat deliverables report accompanying
this update, per the reviewer's explicit ask; this document and
`docs/PHASE_STATUS.md` are updated so the repository itself no longer
presents pre-Tranche-1 Phase 8-12 evidence as trustworthy.

**Two additional instances of the item-3/D baseline bug and the item-13
execution-shortcut were found and fixed during Tranche 1 that weren't in
the original scope list** — `scripts/run_one_step_controller_demo.py` had
its own local `run_baseline_round()` with the identical `spot_prev`/
absolute-`t`/no-depth-walk bugs (its own docstring's whole purpose is
"Compare against baseline on identical replay," so this was a live,
consequential instance, not a dormant one — confirmed by demo output: 0
baseline positions before the fix, 4/4 after), and
`scripts/run_order_supervisor_demo.py`'s controller FAK dispatch had the
same direct-apply shortcut item 13 fixed in `walkforward/ablations.py`/
`shadow/runner.py`. Both fixed identically, using the same shared
`elapsed_t()` helper and `ExecutionSimulator.execute_taker()` already
built for the originally-scoped fixes.

**Not done, and deliberately so** — real profitability, optimal parameter
values, or any ablation "winning" cannot be claimed from this pass. Every
demo re-run in this Tranche is a structural-correctness check (pipeline
executes, causality holds, no-leakage trace passes, accounting
reconciles), explicitly not economic evidence, consistent with item 4's
finding that no synthetic-data claim of that kind was ever supportable.

**Tranche 2 is not started.** Per the reviewer's explicit instruction,
stopping here for review before any proactive-BUFFER_BUILD architecture,
soft-regime redesign, maker cancellation redesign, or chronological
maker-fill work begins.

---

**WAITING FOR APPROVAL (round 2)** before starting the Tranche 1-5
implementation plan above. Every item in Addendum A-L was independently
re-verified against actual code, not accepted on the reviewer's framing
alone — all of A-E and I-L confirmed true (several wider-reaching than
framed), and the one real error found (this document's own original
item 12-13 math) corrected in Addendum F-H. **No implementation changes
have been made in this pass beyond amending this document.** *(Superseded
by the Tranche 1 — Implemented section above; kept for the audit trail.)*

---

# Tranche 1.1 — Implemented

**Approved and implemented**, correcting 8 items found in a re-review of
the pushed Tranche 1 implementation:

1. Round-disjoint q-model/calibration split (`_split_rounds_for_fit_and_calibration`,
   `walkforward/pipeline.py`) — the per-window fit/calibration split
   previously operated on individual examples, not rounds, so a single
   round's own decision-point examples could land partly in the fit set
   and partly in calibration despite sharing one settlement outcome.
2. Parameter sensitivity swept `edge_min` directly on a window's TEST
   rounds in `run_walk_forward_ablation_demo.py` — fixed to use
   `window.validate_round_ids` only.
3. `parameter_stability_across_windows` reused one externally-fit model
   for every window instead of each window's own frozen artifacts —
   fixed via `WindowArtifacts`/`window_artifacts: dict[int, WindowArtifacts]`.
4. `LeakageTrace` was pooled across every window, which could mask or
   false-positive leakage checks once a round legitimately moves from one
   window's TEST to a later window's TRAIN — fixed to one trace per
   window, keyed by `window_index`.
5. `taker_sizing_boundaries`'s risk-budget boundary used the flat
   `max_directional_spend(g_current, g_min)` budget universally, which is
   only exact when the purchased side is already the non-minimum side —
   fixed with `directional_projected_g`, the exact side-aware kernel
   evaluation (`G_U(x)=min(U+x,D)-[C+K_U(x)]`).
6. `run_one_step_controller_demo.py::run_one_step_round` still directly
   converted a chosen FAK candidate into a `Fill`, bypassing
   `ExecutionSimulator.execute_taker` — the one remaining shortcut found
   by an explicit repo-wide audit (all other call sites already fixed in
   Tranche 1).
7. `ExecutionSimulator.execute_taker()` called `submit_taker_order` without
   `asks_at_revalidation` and then resolved synchronously in the same
   call regardless of delay, so a delayed order was resolved against the
   submission book, not the actual book at `matched_ts`. Fixed to accept
   a revalidation book and defer resolution to the caller's own replay
   clock reaching `matched_ts` — see the fuller redesign in Tranche 1.2
   item 2, which replaced this API entirely.
8. Tranche 2 (BUFFER_BUILD, soft regime, dynamic maker, cancel/replace)
   deliberately not started, per instruction.

Full regression coverage, exact test counts, and demo re-run results were
delivered as the chat deliverables report accompanying this update (233 ->
241 tests, all passing).

---

# Tranche 1.2 — Implemented

**Approved and implemented**, a further 12-item correctness/execution-
safety pass found in a re-review of Tranche 1.1:

1. Three more execution paths (`shadow/runner.py`,
   `run_order_supervisor_demo.py`, `walkforward/ablations.py::_run_baseline_round`)
   still called the old `execute_taker`/consumed `.walk` immediately
   regardless of delay — fixed together with item 2's API redesign, using
   a shared `TakerOrderQueue` (`execution/simulator.py`) in every path.
2. `execute_taker`'s design let a caller supply the actual future book
   (`revalidation_asks`) at submission time, so a delayed order's
   eventual fill existed (computed, in the returned object) before
   `matched_ts` ever arrived — "dangerous by construction," since a
   caller reading `.walk` immediately silently bypasses the delay with
   nothing in the API shape to stop it. Redesigned:
   `submit_taker()` -> `PendingTakerOrder` (no future-fill information at
   all for a delayed order) -> `resolve_taker(pending, asks_at_match, now_ts)`
   (the ONLY place a delayed order's real fill is ever computed, and only
   once `now_ts >= matched_ts`).
3. `taker_sizing_boundaries` had no candidate-specific worst-price limit,
   only the shared depth/marginal-edge-only `p_max` — insufficient to
   protect `g_min`/`spend_cap` against an adverse repricing during the
   delay window. Added `taker_max_execution_price` (exact,
   `K_max,s(x) = min(K_s^G(x), K^B)`, inverted to a raw price via
   monotonic binary search, floored to the tick grid) and
   `TakerSizing.max_execution_price_by_qty` (per-quantity, not shared).
4. Regression proving delayed repricing cannot breach `g_min`, using the
   exact named scenario (`p_submit=0.50, p_later=0.80, fee_rate=0.07`).
5. `PendingExposure`/conservative admission: `TakerOrderQueue` now defers
   a second delayed taker submission while one is already
   `PENDING_DELAY`, rather than allowing unbounded concurrent exposure
   with no aggregate accounting.
6. `exposure_from_open_maker_orders` interface added (not yet wired into
   an admission gate at this point — deferred to Tranche 2A item 4).
7. Probability calibration could silently fit Platt on a single-class
   calibration window (the common case at small default window sizes,
   since every example from one round shares that round's one outcome) —
   added `IdentityCalibrator` fallback
   (`UNCALIBRATED_INSUFFICIENT_CLASS_DIVERSITY`) plus backward-expanding
   calibration-window search for class diversity.
8. Audited remaining Phase 8-12 demo/test model constructors for the same
   round-vs-example split issue Tranche 1.1 item 1 fixed in the walk-
   forward pipeline specifically — added `round_ordered_split`
   (`model/walkforward.py`) and swapped every remaining
   `time_ordered_split` call site (5 demo scripts, 2 test fixtures) to it.
9. Fixed the `max_directional_spend()` docstring's own worked example
   ($100 -> $50).
10. Repository hygiene: removed 201 tracked `__pycache__`/`*.pyc` files,
    added `.gitignore`.
11. Walk-forward scope clarification: the ablation matrix (fixed configs)
    and the sensitivity/stability sweep (fixed grid on VALIDATE) are
    explicitly documented as NOT implementing a
    `Config*_i = argmax_theta Metric(VALIDATE_i, theta)` -> evaluate-on-
    `TEST_i` closed selection loop — that belongs to the later real-data
    calibration tranche.
12. Acceptance gate: full suite (241 -> 274 tests), all explicitly-named
    regressions added and passing, demo re-runs clean.

---

# Tranche 2A — Implemented (Aggregate Hard-Risk Envelope)

**Approved and implemented**, correcting the `PendingExposure`
mathematics and completing the aggregate hard-risk admission
architecture Tranche 2B-2E (BUFFER_BUILD, soft regime, dynamic maker,
economic cancel/replace) is gated on:

1. `PendingExposure.worst_case_portfolio()` assumed "every active order
   fills simultaneously" was always the worst case for
   `G = min(U,D) - C` — proven false by the reviewer's own counterexample
   (`U=D=C=0`, pending 100 UP@0.40 and 100 DOWN@0.40: both filling gives
   `G=+20`; only UP filling gives `G=-40` — strictly worse despite fewer
   fills). Replaced with `PendingExposure.worst_case_g`, an exact `min`
   over every fill subset (`2^n` scenario enumeration while
   `n <= DEFAULT_EXACT_SUBSET_CAP`, else a conservative analytical bound
   crediting no protective share contribution from any pending order).
2. Pending taker/maker exposure cost omitted fees — fixed to
   `K_max = x*p_limit + Fee(x, p_limit)`, routed through the same
   `FeeConfig` confirmed fills use, in both `exposure_from_pending_takers`
   and `exposure_from_open_maker_orders`.
3. Added `RiskView` (`portfolio/exposure.py`): combines confirmed
   portfolio + pending taker exposure + open maker exposure into
   `committed_spend`/`worst_case_g`/`potential_up_position`/
   `potential_down_position`, and `admits(candidate, g_min, spend_cap)`
   for hard admission — kept explicitly distinct from the probabilistic
   EV model candidates use for ranking (`ExpectedState != HardRiskState`).
4. `exposure_from_open_maker_orders` existed since Tranche 1.2 but was
   never wired into an actual admission path — now checked via
   `RiskView.admits()` before registering a new maker order with
   `OrderSupervisor`, in both `walkforward/ablations.py`'s
   supervisor-enabled ablation arm and `run_order_supervisor_demo.py`.
5. `ShadowRunner`'s delayed-order resolution book was gated on `recv_ts`
   (this system's own wire-arrival time) — corrected to the default
   `event_time`/`source_ts` gate, since a delayed taker's fill is
   determined by the real exchange's own book at `matched_ts`, which has
   nothing to do with when *this* system happened to receive the update.
   The strategy's own decision-time visibility gate correctly stays on
   `recv_ts` — these are two genuinely different clocks
   (`strategy_view -> recv_ts`, `execution_truth -> event_time`).
6. The per-window calibration split (Tranche 1.2 item 7) only searched
   for calibration-side diversity — could leave the raw model fit on a
   single-class prefix. `_split_rounds_for_fit_and_calibration` now
   searches bidirectionally (both growing AND shrinking the calibration
   window from the default point) for a split where BOTH sides
   independently have `>= 2` classes, returning `None` (no model fit at
   all) when genuinely unreachable rather than a partially-diverse split.
7. Documentation: this section, plus the Tranche 1/1.1/1.2/2A summary
   added to `docs/PHASE_STATUS.md`'s "Notable bugs" section.

Proceeding to Tranche 2B-2E (BUFFER_BUILD/REBALANCE purposes, soft
regime control, dynamic maker price/quantity/TTL, economic hold/cancel/
replace) in the same pass, per instruction. Exact file list, test
counts, and the risk-envelope/BUFFER_BUILD/soft-regime/maker/cancel-
replace architecture summaries are in the chat deliverables report
accompanying this update.

# Tranche 2B-2E — Implemented (proactive geometry, soft regime, dynamic maker, economic cancel/replace)

**Approved and implemented**, completing Tranche 2 on top of 2A's
aggregate hard-risk admission. Full suite: 317 passed, 0 failed.

**2B — BUFFER_BUILD/REBALANCE proactive purposes** (`portfolio/math.py`,
`optimizer/candidates.py`, `optimizer/controller.py`):
- `OrderPurpose` extended `ALPHA, HEDGE` -> `ALPHA, HEDGE, BUFFER_BUILD,
  REBALANCE`. `REBALANCE` is a structural enum member only — the prompt
  gave an exact formula for BUFFER_BUILD but none for REBALANCE, and
  inventing one would be exactly the "don't tune economics on synthetic
  data" instruction this tranche repeats; no generator produces it.
- Exact kernel identities added and proven against the unchanged
  portfolio kernel (`tests/test_portfolio_math.py`):
  `delta_g_directional(u,d,side,x,k_x) = min(u+x,d)-min(u,d)-k_x` (UP) /
  symmetric for DOWN, and independently `delta_ev_directional(side,x,k_x,q)
  = q*x-k_x` (UP) / `(1-q)*x-k_x` (DOWN) — the two are never equated;
  BUFFER_BUILD candidates are accepted on `delta_g > 0` alone and may
  carry negative `delta_ev`, proven by
  `test_buffer_build_generates_candidate_with_positive_delta_g_even_when_delta_ev_is_negative`
  and its mirror-image ALPHA test
  (`test_alpha_candidate_can_have_positive_ev_and_negative_g_independently`).
- `generate_buffer_build_candidates` targets the underrepresented side's
  gap (`x0 = |U-D|`), walks the book with `_risk_boundary_step`'s exact
  unimodal ΔG-vs-x logic to find the ΔG=0 crossing, and offers
  `min(x0, boundary_qty) * 0.999` (a small safety margin off the exact
  boundary) as a single candidate — generated unconditionally of the
  regime's `permitted_actions` (same as HEDGE), gated behind
  `cfg.enable_buffer_build` (default `False`).
- Exempted from `edge_min` (`_EDGE_MIN_EXEMPT_PURPOSES = (HEDGE,
  BUFFER_BUILD)`) since both are geometry-driven, not edge-driven,
  candidates.
- **Bug found and fixed via Hypothesis fuzzing while adding these
  tests** (pre-existing, not introduced by 2B): `_risk_boundary_step`
  divided by zero (`(g_start-g_min)/(g_start-g_kink)`) when floating-point
  underflow at an astronomically small `x0` made `g_kink` numerically
  identical to `g_start`. Fixed with `denom <= 1e-15` guards before both
  interpolation divisions.

**2C — soft regime control** (`optimizer/config.py`, `optimizer/
controller.py`, `optimizer/types.py`):
- `CandidateAction.selection_penalty: float = 0.0` (new field) and
  `OneStepController._family_gate(family, permitted_actions) ->
  (should_generate, penalty)`: with `cfg.soft_regime=False` (default,
  the existing hard-gate ablation, unchanged bit-for-bit — see
  `test_hard_regime_gate_still_produces_no_candidate_for_a_disallowed_family`),
  a disallowed family generates nothing, exactly as before. With
  `cfg.soft_regime=True`, the family is generated anyway, tagged with
  `cfg.regime_prior_penalty` as `selection_penalty`.
- Final selection: `argmax(delta_ev + lambda_g*g_after -
  selection_penalty)` — the penalty is subtracted only from the
  *selection* score, never from `delta_ev` itself, so a large enough edge
  can still beat WAIT despite the penalty
  (`test_soft_regime_can_still_select_a_strong_enough_disfavored_candidate`),
  while a permitted family never carries a penalty
  (`test_permitted_family_never_gets_a_penalty_under_soft_regime`).
- Portfolio-repair/buffer-build candidates are already generated
  independent of `permitted_actions` (2B, and HEDGE before it) — soft
  regime doesn't gate them at all, satisfying "portfolio repair/buffer
  actions must not be blocked by directional seed WAIT state" by
  construction rather than a special case.

**2D — dynamic maker price/quantity/TTL** (`optimizer/candidates.py`,
`optimizer/controller.py`, `optimizer/config.py`):
- `dynamic_maker_sizing(q, side, tau, sigma, portfolio, cfg) -> (qty, ttl,
  offsets)`: quantity is volatility-damped
  (`maker_quantity/(1+sigma)`) and capped at the live risk budget
  (`G-g_min`, the same concept `taker_sizing_boundaries` already uses);
  TTL is volatility-damped and additionally capped at the round's
  remaining time (`tau`); the price-offset grid widens (more ticks
  probed) as volatility rises. An explicit structural placeholder, not
  fit to synthetic PnL (item 37/Addendum J) — the functional form is a
  documented stand-in for Tranche 4 real-data calibration.
- `OneStepController.decide()` gained optional `tau: float = 0.0, sigma:
  float = 0.0` parameters (defaulting to values that are inert unless
  `cfg.dynamic_maker=True`), so no existing caller (`ablations.py`,
  `shadow/runner.py`, `mpc/controller.py`, both demo scripts) needed to
  change. `cfg.dynamic_maker` (default `False`) switches the MAKER_UP/
  MAKER_DOWN blocks between the fixed `maker_quantity`/`maker_horizon_s`/
  `maker_price_offsets_ticks` constants and `dynamic_maker_sizing`'s
  output — proven byte-identical to the fixed path at the default even
  when `tau`/`sigma` are passed
  (`test_dynamic_maker_controller_wiring_is_off_by_default_and_matches_fixed_constants`),
  and proven to actually switch when enabled
  (`test_dynamic_maker_controller_wiring_uses_dynamic_sizing_when_enabled`).
- "No new maker may be posted if any admissible fill subset would violate
  hard risk limits" required no new wiring: 2A's `RiskView.admits()` gate
  in `ablations.py`/`run_order_supervisor_demo.py` runs at *dispatch*
  time on whatever candidate was chosen, dynamically-sized or not — the
  existing `test_ablations_maker_admission_uses_risk_view_and_rejects_when_unsafe`
  already exercises this composition.

**2E — economic hold/cancel/replace with hysteresis**
(`supervisor/predicates.py`, `supervisor/supervisor.py`,
`supervisor/config.py`):
- Removed the unconditional categorical cancel rules for `regime_flip`,
  `edge_failure`, `time_compression`, and the unconditional-if-true
  `book_displacement` replace check. The two genuine safety triggers
  (`feed_stale`, `risk_breach`) remain unconditional, checked first,
  exactly as before.
- Everything else now goes through three value functions
  (`predicates.value_hold/value_cancel/value_replace`) and
  `a* = argmax(V_hold + hysteresis_margin, V_cancel, V_replace)`:
  - `V_hold = current_delta_ev`, degraded by `regime_flip_penalty` if the
    origin regime no longer holds and/or `time_compression_penalty` if
    remaining time makes a passive fill unlikely — a flip lowers the
    order's value, it no longer vetoes it outright.
  - `V_cancel = edge_min - cancel_cost` — the same "expected filled edge"
    floor `edge_failure` used before, now an indifference point rather
    than a hard veto, minus the fixed cost of executing a cancel.
  - `V_replace = current_optimal_ev - churn_threshold` if a new tick's
    EV was actually evaluated this review, else `None` (not a candidate
    action) — compared against the order's *current* held value, not its
    stale submit-time EV (a deliberate correction from the old
    `book_displacement(current_optimal_ev, ev_at_submit, ...)` baseline).
  - All four new coefficients (`regime_flip_penalty`,
    `time_compression_penalty`, `cancel_cost`, `hysteresis_margin`)
    default to `0.0`; at the defaults `V_cancel` winning reduces exactly
    to the old `edge_failure` comparison, so every pre-2E caller's
    default behavior for the *edge* dimension is unchanged. The
    behavioral change specifically targeted by this tranche — a regime
    flip alone no longer forces a cancel — is real at the defaults, since
    `regime_flip_penalty=0.0` means a flip no longer degrades `V_hold` at
    all unless configured to
    (`test_regime_flip_alone_does_not_force_cancellation_when_edge_remains_strong`,
    replacing the old `test_rapid_regime_flip_cancels_immediately` which
    asserted the removed unconditional behavior).
  - Hysteresis: `hysteresis_margin` is an incumbency bonus added to
    `V_hold` before the argmax, so a small, possibly-noisy edge dip does
    not by itself flip the decision away from HOLD
    (`test_hysteresis_margin_prevents_thrashing_on_a_marginal_edge_dip`).
  - CANCEL's `reason` field (used only for journaling/reports) is
    attributed post-hoc by `_cancel_reason` — REGIME_FLIP if the origin
    regime changed, else TIME_COMPRESSION if `tau` is short, else
    EDGE_FAILURE — diagnostic labeling, not a second decision path.
- **Known limitation carried forward**: `walkforward/ablations.py`'s
  supervisor-enabled ablation arm still never passes
  `current_optimal_ev` into `review_order` (a pre-existing gap noted in
  its own comment since Tranche 2A), so `V_replace` is always `None`
  there and REPLACE can never actually fire through that call site —
  only `run_order_supervisor_demo.py` computes a real replacement tick.
  Wiring a real "evaluate the current best alternative tick" step into
  the ablations harness is a separate integration task, not part of the
  policy redesign this tranche specified.

**Files changed (2B-2E only; 2A's files listed in the section above):**
`src/xamarinbot/portfolio/math.py`, `src/xamarinbot/optimizer/
candidates.py`, `src/xamarinbot/optimizer/controller.py`,
`src/xamarinbot/optimizer/config.py`, `src/xamarinbot/optimizer/
types.py`, `src/xamarinbot/supervisor/predicates.py`,
`src/xamarinbot/supervisor/supervisor.py`, `src/xamarinbot/supervisor/
config.py`, `tests/test_portfolio_math.py`, `tests/test_optimizer.py`,
`tests/test_supervisor.py`, this file, `docs/PHASE_STATUS.md`.

**Not done in this pass, by explicit instruction**: Phase 13
(real-capital trading) was not started. No parameter introduced across
2B-2E was fit or selected against this repository's synthetic PnL —
`dynamic_maker_sizing`'s functional form, and the four new
`SupervisorConfig` economic coefficients, are documented structural
placeholders defaulting to values that reduce to prior behavior, not
tuned settings.

# Tranche 2.1 — Implemented (integration and mathematical closure)

**Approved and implemented**, closing 13 integration/math gaps in
Tranche 2's architecture before real-market-data work begins. Full
suite: 340 passed, 0 failed (up from 317 at the end of Tranche 2 -
23 new tests added across items 1-13's acceptance list).

1. **Safety-override/rate-limit ordering** (`supervisor/supervisor.py`):
   `feed_stale`/`risk_breach` are now checked *before*
   `min_action_interval_s`, not after - a stale feed or risk breach
   arriving inside the rate-limit window must still CANCEL immediately.
   Only ordinary economic HOLD/CANCEL/REPLACE churn is throttled.
2. **RiskView as the universal dispatch gate**
   (`portfolio/exposure.py`, `optimizer/candidates.py`,
   `walkforward/ablations.py`, `scripts/run_order_supervisor_demo.py`):
   `RiskView.admits()` gained a `position_limit` parameter (previously
   exposed via `potential_up_position`/`potential_down_position` but
   never enforced) using a plain sum (not the subset envelope - a side's
   position only ever grows as more of its own orders fill, unlike `G`).
   `candidate_exposure()` converts any not-yet-submitted
   `CandidateAction` into the same `ActiveOrderExposure` shape pending/
   open orders already use, and every dispatch site - ALPHA/BUFFER_BUILD/
   HEDGE taker, MAKER, REPLACE - now calls `risk_view.admits()`
   immediately before submission, not just maker placement.
3. **Hard admission before final selection** (`optimizer/controller.py`):
   `OneStepController.decide()` gained an optional `risk_view` parameter;
   when provided, every non-WAIT candidate is checked against it *before*
   `argmax` runs, marking aggregate-risk-unsafe candidates invalid via
   `violated_constraints += ("aggregate_risk",)` rather than only
   rejecting the winner at dispatch with nothing to fall back to - so
   `chosen = argmax_{a in HardRiskAdmissibleActions} J(a)` holds, and a
   rejected top candidate reranks to the next-best legal one automatically.
4. **Purpose-aware taker execution price** (`optimizer/candidates.py`):
   `purpose_aware_max_execution_price()` replaces HEDGE/BUFFER_BUILD's
   `limit_price=1.0` (effectively unconstrained). ALPHA delegates
   unchanged to `taker_max_execution_price` (marginal-edge/g_min/spend
   bounded). BUFFER_BUILD:
   `K_max^buffer(x) = min[min(U+x,D)-min(U,D), min(U+x,D)-C-G_min, B-C]`
   (UP; DOWN symmetric). HEDGE: the same G_min/spend bound without the
   parity term (its quantity is already sized to land at G_min against
   the best-ask assumption; this ceiling guarantees that target survives
   adverse execution).
5. **Breach-recovery semantics** (`optimizer/candidates.py::_finalize`):
   when `G_before < g_min`, ALPHA is prohibited outright regardless of
   its own numbers; every other purpose is admitted precisely when
   `G_after > G_before` (even short of `g_min`) and rejected when
   `G_after <= G_before` - "never allow G_after < G_before" while already
   breached. The ordinary hard rule (`G_after >= g_min`) is unchanged
   when `G_before >= g_min`.
6. **Maker soft-risk weighting corrected** (`optimizer/types.py`,
   `optimizer/candidates.py`, `optimizer/controller.py`):
   `CandidateAction.expected_delta_g` is `ΔG` for a deterministic taker
   fill, `ρ*ΔG` for a fill-probability-weighted maker fill, `0.0` for
   WAIT. Selection is now `J = delta_ev + lambda_g*expected_delta_g -
   selection_penalty` - the prior formula used `c.g_after` (the
   ABSOLUTE, if-filled, and for makers *unweighted* G level) directly in
   an additive score with `delta_ev` (always a marginal quantity), both
   dimensionally and probabilistically wrong. `g_after` itself is
   unchanged and still used for the unconditional hard safety check.
7. **BUFFER_BUILD multi-quantity candidate set**
   (`optimizer/candidates.py::generate_buffer_build_candidates`):
   replaced the single "buy the ΔG-peak quantity" candidate with a
   bounded set - exchange minimum, small-quantity grid, book-depth
   boundaries, spend boundary, `g_min` risk boundary, and the parity
   (`ΔG=0`) boundary - each independently evaluated for `ΔEV(x)`/`ΔG(x)`
   and kept only when `ΔG(x) > 0`, letting the controller's own `argmax`
   choose the quantity actually worth it.
8. **`dynamic_maker_sizing` dimensional bug fixed**
   (`optimizer/candidates.py`, now `dynamic_maker_candidates`
   /`MakerCandidateSpec`/`_maker_feasible_quantity`): the prior draft
   compared `qty` (shares) directly against `portfolio.G - g_min`
   (dollars) via `min()` - dimensionally invalid whenever price != 1.0.
   Replaced with an exact per-PRICE feasible-quantity walk (reusing
   `_risk_boundary_step`'s tested unimodal `G(x)` shape, treating one
   resting maker price as a single "infinite-depth level"), respecting
   `G_min`/`spend_cap`/`position_limit` together. TTL: if `tau` is
   positive but below the new `cfg.min_maker_ttl_s`, no maker candidate
   is generated at all, instead of the prior `max(1.0, ttl)` clamp
   silently proposing a TTL that outlives the round.
9. **REPLACE completed end-to-end** (`optimizer/candidates.py`
   - `ReplacementPlan`/`evaluate_replacement_plan` -,
   `walkforward/ablations.py`, `scripts/run_order_supervisor_demo.py`):
   both integration sites previously called `review_order` with
   `current_optimal_ev=None` always, making REPLACE structurally
   unreachable. `evaluate_replacement_plan` re-evaluates every price on
   the current maker grid (via `evaluate_maker_candidate`) and returns
   the highest-`delta_ev` valid tick; if REPLACE wins, the plan's own
   `ActiveOrderExposure` is checked against a `RiskView` that EXCLUDES
   the order being replaced (about to be torn up) before `apply_replace`
   ever executes.
10. **V_cancel corrected** (`supervisor/predicates.py`): `V_cancel =
    -cancel_cost` only - `edge_min` is a decision THRESHOLD on whether
    holding is worth it, not economic value received from canceling. It
    now lives in a new `hold_eligible(effective_delta_ev, cfg)` gate
    (`effective_delta_ev >= edge_min - hysteresis_margin`, a classic
    control-theory hysteresis band) that determines whether HOLD is even
    a candidate action, rather than being additively folded into any
    V()'s magnitude - the first draft's attempt to fold it into `V_hold`
    instead of `V_cancel` was tried and rejected during this pass (it
    distorted the HOLD-vs-REPLACE comparison for permissive `edge_min`
    configurations, caught by the existing test suite before landing).
11. **Soft regime penalty marked STRUCTURAL/UNCALIBRATED**
    (`optimizer/config.py`): `regime_prior_penalty`'s docstring now
    states explicitly that a flat dollar penalty has quantity-scale
    dependence and is not a calibrated prior, pending real data to
    determine whether the correct representation is per-share,
    probability/logit, or state-dependent.
12. **`lambda_g=0.01` renamed `TEST_ONLY_LAMBDA_G`**
    (`walkforward/ablations.py`): a named module constant, documented as
    an ablation-harness knob selected by observing this repository's own
    synthetic dataset (with the item-6 correction noted as superseding
    the original magnitude analysis), never to be promoted into a
    production `OneStepConfig`.
13. **Acceptance tests added**: `tests/test_supervisor.py` (safety-
    override-bypasses-rate-limit x2, hold_eligible unit tests, V_cancel
    correction), `tests/test_exposure.py` (position_limit enforcement x3,
    combined pending-taker+open-maker admission), `tests/test_optimizer.py`
    (BUFFER_BUILD multi-quantity x2, purpose-aware price ceilings
    survive-adverse-repricing x2, HEDGE never uses limit_price=1.0,
    breach-recovery x4, maker expected_delta_g rho-weighted x2,
    candidate reranking to next-best-legal), `tests/test_walkforward.py`
    (REPLACE actually reaches `apply_replace` and is RiskView-gated,
    integration-level).

**Files changed (Tranche 2.1):** `src/xamarinbot/supervisor/{predicates,
supervisor,config}.py`, `src/xamarinbot/portfolio/exposure.py`,
`src/xamarinbot/optimizer/{candidates,controller,types,config}.py`,
`src/xamarinbot/walkforward/ablations.py`,
`scripts/run_order_supervisor_demo.py`, `tests/{test_supervisor,
test_exposure,test_optimizer,test_walkforward}.py`, this file.

**Known limitation carried forward, scope-noted not silently dropped**:
`RiskView.admits()`'s hard `g_min` check remains a strict floor
regardless of purpose - it does not itself apply Tranche 2.1 item 5's
per-candidate breach-recovery relaxation, since `ActiveOrderExposure`
carries no purpose tag and extending the aggregate envelope to be
purpose-aware would be a materially larger change than requested here.
Breach-recovery is enforced at the single-candidate `_finalize` layer,
which is where every candidate is generated and where the
purpose/before/after state is already available.

**SUPERSEDED by Phase 12B Tranche 2.2 item 1** - the limitation above was
itself a real contradiction (a strict `g_min` floor made every recovery
candidate aggregate-invalid once already breached, since the envelope's
own empty-fill subset reproduces the pre-candidate G exactly). `admits()`
now takes an `is_recovery_candidate` flag and a two-mode envelope - see
the Tranche 2.2 section below.

# Tranche 2.2 — Implemented (final synthetic correctness closure)

**Approved and implemented**, closing six remaining integration/math
gaps before synthetic strategy work stops entirely and real-market-data
collection begins. Full suite: 356 passed, 0 failed (up from 340 at the
end of Tranche 2.1 - 16 new tests added across items 1-6's acceptance
list).

1. **Breach-recovery vs `RiskView.admits()` contradiction fixed**
   (`portfolio/exposure.py`, `optimizer/candidates.py`): `admits()` now
   computes `worst_g_base` (before the candidate) and `worst_g_new`
   (with it) separately. SAFE mode (`worst_g_base >= g_min`): unchanged,
   `worst_g_new >= g_min` required. RECOVERY mode (`worst_g_base <
   g_min`): a new `is_recovery_candidate` flag (`is_recovery_purpose()`,
   the same HEDGE/BUFFER_BUILD set `_finalize`'s own breach-recovery
   check uses) - non-recovery (ALPHA) candidates rejected outright;
   recovery candidates admitted iff `worst_g_new >= worst_g_base` (never
   worsens the aggregate envelope), matching `_finalize`'s own `G_after >
   G_before` rule. The exact reviewer counterexample (`G_current=-50,
   g_min=-20`, a hedge improving to `-25`) is now admissible - proven
   directly (`test_risk_view_admits_recovery_candidate_when_it_does_not_worsen_the_aggregate_envelope`)
   and via a full integrated-controller test with a real `RiskView`
   (`test_breach_recovery_candidate_survives_a_real_risk_view_and_beats_wait_when_partial`).
   The SAME contradiction existed in `purpose_aware_max_execution_price`'s
   HEDGE/BUFFER_BUILD price-ceiling formula (Tranche 2.1 item 4) -
   `K_max` was always negative once breached, since it was computed
   against the unreachable `g_min` too; fixed with the same `g_floor =
   g_min if G_before >= g_min else G_before` substitution.
2. **MPC risk/utility integration** (`mpc/controller.py`):
   `MPCController.decide()` gained an optional `risk_view` parameter,
   forwarded into the underlying `OneStepController.decide()` call for
   the immediate action - the action MPC ultimately returns is already
   aggregate-risk-admissible, not merely rejected by dispatch afterward.
   A new `candidate_selection_score(candidate, cfg)` in
   `optimizer/candidates.py` (`ΔEV + lambda_g*expected_delta_g -
   selection_penalty`) is now the ONE function both
   `OneStepController`'s own `argmax` and every value MPC computes from a
   `CandidateAction` (`sequence_values`, `_rollout`'s per-level value) go
   through - the prior draft's `sequence_value = candidate.delta_ev`
   silently ignored `lambda_g`/`selection_penalty` entirely. Deeper
   hypothetical rollouts (`horizon_steps > 1`) have no exposure-transition
   model of their own; per the prompt's own "prefer the simpler safe
   solution," `decide()` falls back to the plain risk-aware one-step
   decision (`used_fallback=True`) whenever `risk_view` reports real
   active exposure (pending takers or open makers), rather than exploring
   hypothetical future states with no model for how that exposure
   evolves. MPC remains labeled experimental/non-production in its own
   module docstring until real data validates the deeper rollout.
   `walkforward/ablations.py`'s MPC branch (#8) now also passes the same
   shared `risk_view` used by every other dispatch path in that harness.
3. **`TradingSession` - one shared execution/session component**
   (new `src/xamarinbot/execution/session.py`): owns confirmed
   `PortfolioState`, `TakerOrderQueue`, open maker orders (via
   `OrderSupervisor`), `RiskView` construction, and every submit/cancel/
   replace/expire operation - `risk_view()`, `resolve_ready_takers()`,
   `review_open_orders()` (with an optional `on_decision` callback for
   journaling), `dispatch()`. **`ShadowRunner` is fully migrated onto
   it** - it previously had no `RiskView`, no aggregate maker exposure,
   no `OrderSupervisor`, and resolved every maker candidate via an
   immediate Bernoulli draw at submission instead of tracking it as a
   genuinely open, reviewable, cancelable/replaceable order. Now it
   constructs a `RiskView` every decision, passes it into
   `OneStepController`, RiskView-admits every dispatch, tracks open
   makers via `OrderSupervisor`, runs supervisor review on later
   decision points, and only ever applies a taker fill at its own
   resolved `matched_ts` - exactly the walk-forward supervisor-enabled
   ablation arm's own execution stack. `scripts/run_order_supervisor_demo.py`
   is migrated onto the same class. `walkforward/ablations.py`'s
   `_run_controller_round` was **not** rewritten onto this specific class
   in this pass (a materially larger, riskier restructure of a loop with
   8 ablation-matrix-specific behavioral branches - supervisor-vs-not,
   baseline-vs-controller, one-step-vs-MPC - all covered by 300+ existing
   tests) but already shares every one of `TradingSession`'s own
   underlying primitives (`ExecutionSimulator`, `TakerOrderQueue`,
   `OrderSupervisor`, `RiskView`, `evaluate_replacement_plan`) - no
   execution/risk logic is duplicated there, only loop/dispatch glue.
4. **BUFFER_BUILD exact-minimum bug fixed**
   (`optimizer/candidates.py::generate_buffer_build_candidates`): the
   0.999 float-safety margin was applied to EVERY candidate quantity
   uniformly, turning `raw_qty=taker_min_size=1.0` into `qty=0.999`,
   which then failed its own `>= taker_min_size` check - silently
   discarding the exact-exchange-minimum candidate whenever it was the
   binding one. Quantities are now split into `exact_quantities`
   (exchange minimum, interior small-quantity-grid points, depth points
   strictly below the feasible ceiling - never margined) and
   `boundary_quantities` (the parity/risk/spend ceiling itself - margined
   by 0.999, as before). A candidate at exactly the exchange minimum
   survives whenever its own `ΔG > 0`.
5. **`tau=0` semantic ambiguity fixed** (`optimizer/candidates.py`,
   `optimizer/controller.py`): `tau` is now `float | None` throughout the
   dynamic-maker path - `None` means "not supplied / unknown" (inert, no
   time constraint applied, matching every existing caller that never
   passed it); a real `tau=0.0` now means "no remaining time at all" and
   suppresses maker generation entirely, no longer silently conflated
   with "unknown" by both defaulting to and special-casing `0.0`. TTL
   computation reordered (volatility damping + 1.0 floor, THEN the `tau`
   cap applied last) so `TTL <= tau` holds unconditionally whenever `tau`
   is known, even in the case where the floor would otherwise push it
   above a small positive `tau`.
6. **Hysteresis contract pinned down explicitly**
   (`tests/test_supervisor.py`): a new test computes `value_hold`/
   `value_cancel` directly and asserts HOLD is still chosen even though
   `V_hold`'s own raw value is strictly less than `V_cancel`'s - stating
   the contract explicitly (hysteresis widens `hold_eligible`'s
   threshold, tolerating a "slightly inferior" HOLD within the configured
   band; it does not make `V_hold` numerically competitive with
   `V_cancel`) rather than leaving the prior test's outcome-only
   assertion to imply it accidentally.

**Files changed:** `src/xamarinbot/portfolio/exposure.py`,
`src/xamarinbot/optimizer/candidates.py`, `src/xamarinbot/mpc/
controller.py`, `src/xamarinbot/optimizer/controller.py` (new
`risk_view`/`tau: float | None` semantics already present from Tranche
2.1, `is_recovery_candidate` wiring added here), new
`src/xamarinbot/execution/session.py`, `src/xamarinbot/shadow/runner.py`
(rewritten), `scripts/run_order_supervisor_demo.py` (rewritten),
`src/xamarinbot/walkforward/ablations.py` (MPC `risk_view` wiring only),
`tests/{test_exposure,test_optimizer,test_mpc,test_shadow,
test_supervisor}.py`, this file, `docs/PHASE_STATUS.md`.

**Remaining limitations**: `walkforward/ablations.py`'s
`_run_controller_round` was not migrated onto `TradingSession` in this
pass (see item 3 above - a scope-limiting decision, not an oversight).
MPC's deeper rollout (`horizon_steps > 1`) still has no model for how
active exposure evolves across hypothetical future states - it avoids
the problem via the "fall back to one-step" rule rather than solving it;
MPC stays experimental/non-production. No parameter introduced across
this pass was fit or selected against this repository's synthetic PnL.

Per the reviewer's explicit instruction, synthetic strategy development
stops here. The next stage is a real market recorder and event-level
shadow dataset against actual Polymarket BTC 5-minute markets, from which
`q`, `rho`, `q_fill`, slippage, latency, and adverse-selection can
finally be estimated from real data rather than further synthetic
tuning.
