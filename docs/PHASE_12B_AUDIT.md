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

**Proposed fix**: Add `BUFFER_BUILD` and `REBALANCE` to `OrderPurpose`.
Implement the pair-buffer identity directly:
`DeltaG_pair(x) = x - K_U(x) - K_D(x)` for acquiring `x` on the currently
cheaper/under-represented side; generate a `BUFFER_BUILD` candidate
whenever `DeltaG_pair(x) > 0` is achievable at some feasible `x`,
independent of current `G` vs `g_min` — i.e. it competes on the same
`argmax` footing as ALPHA candidates via its own `delta_ev`/`delta_g`,
not gated behind a risk-floor breach. Evaluate this across *sequential*
fills at different times (maintaining cumulative `U, D, C` — no
requirement that `UP + DOWN < 1` simultaneously), not only simultaneous
quotes, per item 13's explicit instruction.

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

# Summary (per the prompt's required closing sections)

## Which issues are confirmed

**CONFIRMED BUG** (real defects, not design tradeoffs):
- Item 3 — baseline `spot_prev` bug in `walkforward/ablations.py`, invalidates every Phase 11 baseline comparison to date.
- Item 7 — taker candidate sizing has no intermediate/marginal/risk-budget-derived quantities, only raw depth-level cumulative sums; already reproduced empirical consequence on record (ablation #6).
- Item 9 — `spend_cap` checked per-order, not cumulatively per round.
- Item 10 — favored-side inference uses payoff geometry (`Pi_U >= Pi_D`), not prediction; currently dormant (`p_min` never configured) but would misfire immediately if enabled.
- Item 18 — Phase 11 ablations 7/8 call the supervisor with a hardcoded `current_ev_after=0.0` and the wrong (`unconditional` vs `if-filled`) `G` value; `current_optimal_ev` never passed at all, so `REPLACE` can never fire in that harness.

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
- No other claim in the prompt was found to be factually incorrect
  against the current code — every other numbered item's core technical
  claim was verified true by direct inspection, not merely plausible.

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

Following the prompt's own item-35 ordering exactly, mapped to this
audit's findings:

1. ~~`docs/PHASE_12B_AUDIT.md`~~ — this document.
2. Fix baseline `spot_prev` (item 3) — isolated, low-risk, mirrors
   existing correct pattern in the same file.
3. Fix `spend_cap` cumulative semantics (item 9).
4. Rename/clarify `ev_after` → `delta_ev` (item 5) — mechanical, wide
   diff (~15 files), do before other optimizer changes so they're
   written against the clarified name.
5. Split marginal edge from total EV (item 6).
6. Risk/depth/marginal-edge-aware taker quantity optimization (item 7),
   wiring in the currently-dead `max_directional_spend`.
7. Derive real worst-price protection (item 8, backtest-relevant half);
   defer the dollar-amount translation layer to step 26 (adapter work).
8. Make favored side explicit (item 10).
9. Implement `BUFFER_BUILD`/`HEDGE` as genuinely proactive candidates
   (items 12-13) — the single largest change in this plan.
10. Make maker price/quantity/TTL dynamic (item 15).
11. Correct maker probability/risk utility weighting consistency (item 16).
12. Redesign regime gating as soft prior + hard-gated ablation flag (item 14).
13. Redesign cancellation with `V_hold/V_cancel/V_replace` + hysteresis (item 17).
14. Fix Phase 11 supervisor placeholder evaluation (item 18) — mirror the
    already-correct pattern in `scripts/run_order_supervisor_demo.py`.
15. Implement chronological maker fills (item 19) — blocked on step 18's
    real recorder existing; sequence after step 18 below.
16. Complete real Polymarket/RTDS/Chainlink/user-stream adapters (item 21) —
    token mapping, Chainlink window confirmation, fee-config wiring (item 23),
    pre-round history buffer (item 24).
17. Build the real round recorder (item 31).
18. Run live non-trading data collection (item 31, continued).
19. Rerun model calibration and ablations on real data (items 26, 32) —
    only meaningful once step 18 has produced enough real rounds.
20. Build a genuine event-driven live loop (item 20) and run true
    event-driven shadow trading (item 30 stays deferred; MPC expansion
    does not happen in this pass).
21. Produce the Phase 12B economic report (item 33) with the requested
    Pair/Buffer/Directional/Fees/Slippage/AdverseSelection decomposition.
22. Only then propose Phase 13.

Every step keeps existing passing tests green (item 36's "no existing
test may be deleted simply because the architecture changes" rule);
semantics-changing steps (1, 4, 9, 12, 13, 17 above) get their old test
updated in place with an inline comment explaining why, alongside new
tests, not a silent deletion.

## Files that will change

`optimizer/candidates.py`, `optimizer/controller.py`, `optimizer/config.py`,
`optimizer/types.py`, `portfolio/math.py`, `baseline/config.py` (new
sub-cursor wiring only, not `baseline/strategy.py` itself),
`walkforward/ablations.py`, `supervisor/supervisor.py`,
`supervisor/predicates.py`, `supervisor/config.py`, `regime/matrix.py`
(soft-prior mode, additive), `execution/simulator.py` (once real data
exists), `shadow/runner.py`, `feeds/polymarket_clob.py`,
`feeds/chainlink_twap.py`, `feeds/polymarket_user.py`, a new `recorder/`
module, a new `reports/economic_report.py`, plus the corresponding test
files for every item above and `docs/PHASE_STATUS.md`.

## Mathematical invariants that will remain untouched

- `Pi_U = U - C`, `Pi_D = D - C`, `G = min(Pi_U, Pi_D)`, `R = Pi_U - Pi_D`
  (item 1 — confirmed correct, not touched).
- The `EV_after_total - EV_before_total == q*ΔU + (1-q)*ΔD - ΔC` identity
  (item 5 — the *value* stays; only the *name* changes to `delta_ev`, and
  a test will pin the identity down explicitly for the first time).
- `DeltaG_pair(x) = x - K_U(x) - K_D(x)` (item 13 — new code, but the
  identity itself is exact and testable, not an approximation).
- Causal replay: `event_time <= decision_time` (Phase 2) and
  `recv_ts <= decision_time` (Phase 12's stricter live gate) both stay
  exactly as implemented — no item in this audit calls either into
  question.
- Every existing Hypothesis property test in `tests/test_portfolio_math.py`,
  `tests/test_optimizer.py`, and `tests/test_execution.py` continues to
  hold under the proposed changes (none of the proposed fixes alter the
  fill-simulation or constraint-evaluation math itself, only what
  candidates are generated and what values are compared against
  constraints).

---

**WAITING FOR APPROVAL** before starting the implementation plan above,
per the prompt's explicit instruction. No implementation changes were
made in this pass beyond writing this document.
