#!/usr/bin/env python3
"""Does relaxing the staleness caps buy more skipping, and does it cost accuracy?

THE PRE-REGISTERED QUESTION
findings/prediction_staleness_sweep.md predicted that the staleness caps are
the binding limiter on TAVS's skip rate, and named two falsifiers:

  F1  a skip rate >= ~50% in the no-expiry (40/40) arm would mean the caps
      WERE the limiter and the weight budget is not the next ceiling;
  F2  a skip rate essentially unchanged from ~32% would mean NEITHER cap was
      the limiter -- most likely trust never reaching theta_high.

WHAT THIS SCRIPT MEASURES, AND WHY EACH PIECE
  GATE 0  one variable at a time. The arms must differ from the baseline in
          the two cap fields and nothing else, or no comparison is valid.
  GATE 1  did the cap actually stop firing? `num_forced_stale` is logged, so
          this is measured, not assumed. A client that has NEVER been
          verified is stale by definition, so its first-ever appearance is
          counted as forced-stale at any cap setting -- that floor is exactly
          one event per distinct client and must be subtracted before the
          gate means anything. An earlier version of this script did not
          subtract it, read the resulting 12.5% as "cap still firing", and
          declared the whole sweep uninterpretable. 12.5% was 50 clients over
          400 client-rounds: the floor, not the cap.
  TEST 1  skip rate, paired per seed, with a CI. This is the quantity the
          prediction is about.
  TEST 2  the falsifiers, evaluated as written.
  TEST 3  what DOES gate promotion. Tier 3 requires BOTH t_eff >= theta_high
          AND clean_streak >= k_trust. Both are replayed from the logs and
          each blocked client-round is attributed to one cause.
  TEST 4  accuracy on all four endpoints with CIs and a batch-split check,
          because "no difference" without a CI is not a result.
  TEST 5  WHY the null: the budget absorbs what the cap releases, as a
          bookkeeping identity across the arms.

CAVEAT CARRIED FROM THE SWEEP SCRIPT: byzantine_fraction is 0 in every run
here. The staleness cap is an anti-drift / anti-Sybil guard. Costing nothing
WITHOUT an attacker says nothing about its value WITH one.
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.analysis.oracle.dose_and_schedule import (  # noqa: E402
    ARM, ENDPOINTS, batch_agreement, ci_halfwidth, config_diffs, fmt_row,
    header, jitter, load_seeds, paired_t, t_crit,
)

RESULTS = ROOT / "results"
REF = RESULTS / "sweep_a10_twagg"            # caps 4 / 10, the current default
REF_LABEL = "4 / 10   (baseline)"
ARMS = [
    ("8 / 20   (double)", RESULTS / "stale_a10_sa08_sr20"),
    ("12 / 30  (triple)", RESULTS / "stale_a10_sa12_sr30"),
    ("40 / 40  (no expiry)", RESULTS / "stale_a10_sa40_sr40"),
]
NO_EXPIRY = "40 / 40  (no expiry)"
EXPECTED_DIFFS = {"s_max_appearances", "s_max_rounds"}
SCORE_STREAK_CUTOFF = 0.8      # scheduler resets the streak at or below this


def load_runs(root: Path):
    """{seed -> parsed pipeline_results.json}. Raises on a duplicate seed.

    A duplicate means two config subdirectories are present, which is how a
    rerun under a changed tag silently swaps half a series for a different
    configuration. load_seeds raises on this too; both paths must.

    The seed is matched with the SAME regex load_seeds uses. Splitting on
    "seed" and calling int() on the remainder instead made the two loaders
    disagree about which directories exist: a rerun left as
    `tavs_skip_seed1_rerun` is folded in by load_seeds (so it would enter the
    accuracy table) but killed int() here with an opaque ValueError, taking
    out every other test in the script. One regex, one file set.
    """
    out = {}
    for p in sorted(root.glob(f"**/{ARM}_seed*/pipeline_results.json")):
        m = re.search(rf"{ARM}_seed(\d+)", p.parent.name)
        if not m:
            continue
        seed = str(int(m.group(1)))      # seed01 and seed1 are the same seed
        if seed in out:
            raise RuntimeError(
                f"duplicate seed {seed} under {root}: {p}\nTwo configurations "
                f"are present; point the script at one config subdirectory."
            )
        out[seed] = json.loads(p.read_text())
    return out


def tavs_config(run):
    """The scheduler's own config block.

    It is nested under config['tavs_config']; reading the cap fields off the
    top-level config dict returns None for every one of them, which an
    earlier version of this script printed as the baseline's caps.
    """
    return run["config"]["tavs_config"]


def distinct_clients(run):
    """Number of distinct clients ever sampled -- the forced-stale floor."""
    seen = set()
    for cats in run["round_assignments"].values():
        for v in cats.values():
            if isinstance(v, list):
                seen.update(v)
    return len(seen)


def rates(run):
    """Per-client-round rates for one run.

    forced_stale_net subtracts the first-appearance floor, so it is the rate
    at which the CAP itself fired rather than the rate at which anything was
    stale.

    THE FLOOR EXISTS ONLY WHEN THE BOOTSTRAP IS ON. is_stale() reports a
    never-verified client as stale only when bootstrap_verify_new_clients is
    True; with the ablation flag off it returns False and the client falls
    through to the tier branches instead (landing in Tier 1, since
    initial_trust 0.25 is under theta_low 0.3). Subtracting a floor that is
    not there made forced_stale_net negative -- measured at -0.500 on a
    bootstrap-off fixture whose raw rate was 0.000 -- so the floor is now
    conditional on the flag.

    Returns None -- not {} -- for a run with no cohort, because every caller
    indexes the result by key. Returning {} turned one truncated run into a
    bare `KeyError: 'skip'` that aborted the whole comparison; load_seeds
    deliberately skips such a run instead, and this now matches it.
    """
    h = run.get("scheduling_history") or []
    cohort = sum(r["cohort_size"] for r in h)
    if not cohort:
        return None
    fs = sum(r["num_forced_stale"] for r in h)
    floor = (distinct_clients(run)
             if tavs_config(run).get("bootstrap_verify_new_clients", True)
             else 0)
    return {
        "skip": sum(r["num_promoted"] for r in h) / cohort,
        "verified": sum(r["num_verified"] for r in h) / cohort,
        "forced_stale_raw": fs / cohort,
        "forced_stale_net": (fs - floor) / cohort,
        "bootstrap": bool(
            tavs_config(run).get("bootstrap_verify_new_clients", True)),
    }


def replay_schedule(run):
    """Replay schedule_verifications -- BOTH phases -- and attribute every
    sampled client-round to the branch that actually decided it.

    WHY THIS REPLACED AN EARLIER, WRONG ATTRIBUTION
    A first version treated `t_eff >= theta_high AND clean_streak >= k_trust`
    as the condition for being skipped, and reported everything failing it as
    "blocked". That is the Tier THREE condition, not the condition for
    skipping. The scheduler's Phase 1 is:

        if is_stale(c)            -> V      (hard expiry; never-verified too)
        elif t_eff < theta_low    -> V      (Tier 1)
        elif t_eff >= theta_high and streak >= k_trust:
                                     V if the decoy roll hits else P   (Tier 3)
        else                      -> P      (Tier 2, TENTATIVE)

    Tier 2 is the final `else` and it PROMOTES. So a client under
    theta_high, or with a short streak, is Tier 2 and gets skipped anyway --
    it is not blocked at all. The old table's two headline "gates" (trust bar
    140 client-rounds, streak 81) were counting Tier-2 clients that the
    scheduler was happily skipping. This file's own
    test_trust_by_round_logging docstring already said Tier 2 is tentatively
    promoted; the attribution contradicted it.

    Phase 2 is then the weight budget: sort the tentative P by ascending
    effective trust and demote the lowest until
    gamma = sum_p / (|V| + sum_p) <= gamma_budget.

    FIDELITY. The replay reconstructs trust from the round-keyed log but
    replays the streak and both staleness clocks, so it is close but not
    exact: it reproduces the promoted COUNT to within ~2% while matching the
    exact promoted SET in about a quarter of rounds. Counts are what the
    attribution reports; `set_match` is returned so that limit stays visible
    rather than implied.

    Returns per-seed counts, or None when either log the replay needs is
    absent (see the notes on trust_by_round and oracle_signal_history below).
      - trust_by_round empty: the run predates the logging fix (the baseline
        does), so no round-keyed trust exists.
      - oracle_signal_history empty: no behaviour score exists, so no streak
        can be replayed. Left unguarded every streak stays 0, which silently
        pushes Tier-3 clients down into Tier 2 and misreports the decoy and
        budget columns.
    """
    tc = tavs_config(run)
    tbr = run.get("trust_by_round") or {}
    osh = (run.get("final_trust_state") or {}).get(
        "oracle_signal_history") or {}
    if not tbr or not osh:
        return None
    lo, hi = (tc["pretrain_loss_var_rescale_lo"],
              tc["pretrain_loss_var_rescale_hi"])
    th_lo, th_hi = tc["theta_low"], tc["theta_high"]
    k, tau = tc["k_trust"], tc.get("tau_ramp", 5.0)
    sa, sr = tc["s_max_appearances"], tc["s_max_rounds"]
    cl, budget = tc.get("c_lambda", 8.0), tc["gamma_budget"]
    bootstrap = tc.get("bootstrap_verify_new_clients", True)
    ra = run["round_assignments"]
    det = {str(x["round"]): set(x.get("detected") or [])
           for x in (run.get("byzantine_detection_history") or [])}

    join = {}
    for rk in sorted(ra, key=int):
        for c in (ra[rk].get("sampled") or []):
            join.setdefault(c, int(rk))

    def posterior(t):
        return 1.0 / (1.0 + math.exp(-cl * (t - 0.5)))

    out = {"sampled": 0, "rounds": 0, "logged_promoted": 0,
           "stale": 0, "tier1": 0, "decoy": 0, "tier2_tentative": 0,
           "tier3_promoted": 0, "demoted_by_budget": 0, "replay_promoted": 0,
           "verifications": 0, "streak_reset": 0, "reset_by_score": 0,
           "reset_by_detector": 0, "left_at_k": 0,
           "set_match": 0, "skipped_no_trust": 0,
           "rounds_over_budget": 0, "rounds_empty_pool": 0}
    gamma_pre, gamma_post = [], []
    streak, app, lastv = {}, {}, {}
    for rk in sorted(ra, key=int):
        r, cats = int(rk), ra[rk]
        S = cats.get("sampled") or []
        logged_P = set(cats.get("promoted") or [])
        ver = set(cats.get("verified") or [])
        decoy = set(cats.get("decoy") or [])
        if rk not in tbr:
            out["skipped_no_trust"] += 1
        else:
            out["rounds"] += 1
            out["sampled"] += len(S)
            out["logged_promoted"] += len(logged_P)
            # ---- PHASE 1 ----
            V, P, eff = set(), set(), {}
            for c in S:
                t = min(tbr[rk].get(c, 0.0),
                        1.0 - math.exp(-(r - join.get(c, r)) / tau))
                eff[c] = t
                never = c not in lastv
                # The never-verified case is stale only under the bootstrap;
                # with the ablation flag off, is_stale falls through to the
                # s_max checks, which a never-verified client trivially
                # passes, so it reaches the tier branches instead.
                if ((never and bootstrap)
                        or (not never and (app.get(c, 0) >= sa
                                           or (r - lastv[c]) >= sr))):
                    V.add(c)
                    out["stale"] += 1
                elif t < th_lo:
                    V.add(c)
                    out["tier1"] += 1
                elif c in decoy:
                    # Read the LOGGED decoy set before the tier test. The
                    # decoy draw is ground truth; the replayed tier is not.
                    # Testing tier first meant a client the scheduler really
                    # drew as a decoy, but whose replayed streak fell one
                    # short, was counted as Tier 2 AND entered the tentative
                    # pool -- reported as skipped when reality verified it.
                    V.add(c)
                    out["decoy"] += 1
                elif t >= th_hi and streak.get(c, 0) >= k:
                    P.add(c)
                    out["tier3_promoted"] += 1
                else:
                    P.add(c)
                    out["tier2_tentative"] += 1
            # ---- PHASE 2: the weight budget ----
            def gamma_of(Pset, Vset):
                sp = sum(posterior(eff[c]) for c in Pset)
                z = len(Vset) + sp
                return (sp / z) if z > 0 else 0.0
            # Record gamma_pre only for rounds whose tentative pool is
            # non-empty. A round with nothing promoted has sum_p = 0, so its
            # gamma is a structural 0 and the budget cannot bind there;
            # averaging those in understated the figure (0.775 against 0.816
            # over 400 rounds, 20 of them empty) and would flip a borderline
            # verdict the unconservative way on a looser budget.
            if P:
                g_pre = gamma_of(P, V)
                gamma_pre.append(g_pre)
                if g_pre > budget:
                    out["rounds_over_budget"] += 1
            else:
                out["rounds_empty_pool"] += 1
            order = sorted(P, key=lambda c: eff[c])
            while P and gamma_of(P, V) > budget:
                c = order.pop(0)
                P.discard(c)
                V.add(c)
                out["demoted_by_budget"] += 1
            if P:
                gamma_post.append(gamma_of(P, V))
            out["replay_promoted"] += len(P)
            if P == logged_P:
                out["set_match"] += 1
        # ---- advance trust clocks and the streak, every round ----
        for c in S:
            if c in ver:
                app[c] = 0
                lastv[c] = r
                e = (osh.get(rk) or {}).get(c) or {}
                v = e.get("pretrain_loss_var")
                if v is None:
                    continue
                b = 1.0 - max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
                out["verifications"] += 1
                flagged = c in det.get(rk, set())
                if flagged or b <= SCORE_STREAK_CUTOFF:
                    streak[c] = 0
                    out["streak_reset"] += 1
                    out["reset_by_detector" if flagged
                        else "reset_by_score"] += 1
                else:
                    streak[c] = streak.get(c, 0) + 1
                if streak.get(c, 0) >= k:
                    out["left_at_k"] += 1
            elif c in logged_P:
                app[c] = app.get(c, 0) + 1
    out["gamma_pre"] = statistics.mean(gamma_pre) if gamma_pre else float("nan")
    out["gamma_post"] = (statistics.mean(gamma_post) if gamma_post
                         else float("nan"))
    out["gamma_pre_max"] = max(gamma_pre) if gamma_pre else float("nan")
    return out

def mean_rate(runs, key):
    """Mean of one rate over runs, skipping any run rates() cannot measure."""
    vals = [r[key] for r in (rates(x) for x in runs) if r]
    return statistics.mean(vals) if vals else float("nan")


def _paired(ref_runs, arm_runs, key):
    """Paired per-seed difference, dropping a seed either side cannot measure.

    A seed whose run has no cohort (truncated or crashed) is dropped rather
    than indexed, which is what turned one bad run into a bare
    `KeyError: 'skip'` that aborted the entire comparison.
    """
    out = []
    for s in sorted(set(ref_runs) & set(arm_runs), key=int):
        a, b = rates(arm_runs[s]), rates(ref_runs[s])
        if a and b:
            out.append(a[key] - b[key])
    return out


def section(title):
    print(f"\n{title}\n{'=' * len(title)}")


def main():
    if not REF.exists():
        print(f"missing baseline {REF}")
        return False
    if [n for n, p in ARMS if not p.exists()]:
        print(f"missing arms: {[n for n, p in ARMS if not p.exists()]}")
        return False

    ref_runs = load_runs(REF)
    ref_seeds = load_seeds(REF)
    arms = {n: load_runs(p) for n, p in ARMS}

    section("GATE 0  one variable at a time")
    ok = True
    for name, path in ARMS:
        d = config_diffs(REF, path)
        bad = [x for x in d if x.split(":")[0] not in EXPECTED_DIFFS]
        print(f"  {name:<24} {len(d)} diff(s); unexpected: {bad or 'none'}")
        if bad or not d:
            ok = False
    rc = tavs_config(next(iter(ref_runs.values())))
    print(f"  baseline caps: s_max_appearances={rc['s_max_appearances']}, "
          f"s_max_rounds={rc['s_max_rounds']}; "
          f"theta_high={rc['theta_high']}, k_trust={rc['k_trust']}, "
          f"gamma_budget={rc['gamma_budget']}")
    if not ok:
        print("  FAIL -- arms are not comparable; stopping.")
        return False
    print("  PASS -- only the two cap fields differ.")

    section("GATE 1  did the cap actually stop firing?")
    print("  forced-stale(raw) counts every stale client-round, including each")
    print("  client's first-ever appearance, which is stale by definition at")
    print("  ANY cap. forced-stale(net) subtracts that floor.")
    print(f"\n  {'arm':<24} {'skip':>7} {'verified':>9} "
          f"{'f-stale raw':>12} {'f-stale net':>12}")
    print("  " + "-" * 68)
    for label, runs in [(REF_LABEL, ref_runs)] + list(arms.items()):
        rv = runs.values()
        print(f"  {label:<24} "
              f"{mean_rate(rv, 'skip'):>6.1%} "
              f"{mean_rate(rv, 'verified'):>8.1%} "
              f"{mean_rate(rv, 'forced_stale_raw'):>11.1%} "
              f"{mean_rate(rv, 'forced_stale_net'):>11.1%}")
    net40 = mean_rate(arms[NO_EXPIRY].values(), "forced_stale_net")
    print(f"\n  no-expiry net rate = {net40:.2%}  ", end="")
    print("PASS (cap never fires; the arm tests what it claims)"
          if abs(net40) < 0.01 else "FAIL (cap still firing)")
    if abs(net40) >= 0.01:
        return False

    section("TEST 1  does the skip rate move?  (paired per seed)")
    print(f"  {'arm':<24} {'mean d':>9} {'95% CI':>9} {'t':>7} {'up/n':>7}"
          f"  verdict")
    print("  " + "-" * 68)
    for name, runs in arms.items():
        diffs = _paired(ref_runs, runs, "skip")
        m, t, n = paired_t(diffs)
        sig = (not math.isnan(t)) and abs(t) > t_crit(n)
        print(f"  {name:<24} {m:>+8.1%} {ci_halfwidth(diffs):>8.1%} "
              f"{t:>+7.2f} {sum(1 for d in diffs if d > 0):>4}/{n:<2}  "
              f"{'MOVED' if sig else 'no detectable change'}")

    section("TEST 2  the pre-registered falsifiers")
    sk_ref = mean_rate(ref_runs.values(), "skip")
    sk40 = mean_rate(arms[NO_EXPIRY].values(), "skip")
    print(f"  baseline skip rate (caps  4/10) : {sk_ref:.1%}")
    print(f"  no-expiry skip rate (caps 40/40): {sk40:.1%}"
          f"   ({(sk40 - sk_ref) * 100:+.1f} points)")

    # F2 is conditional on gamma still not binding, so measure it -- BEFORE
    # the demotion loop. Reading gamma after the loop is circular: the loop
    # runs until gamma <= gamma_budget, so the post value is <= the cap by
    # construction no matter how hard the budget bit. An earlier version of
    # this script measured only the post value, saw 0.268 against a 0.35 cap,
    # and concluded the budget was slack. It is not: pre-demotion gamma is
    # 0.775 and the loop demotes about 200 client-rounds per seed.
    budget = tavs_config(next(iter(arms[NO_EXPIRY].values())))["gamma_budget"]
    print(f"\n  the weight budget (gamma_budget = {budget}), measured BEFORE")
    print("  the demotion loop as well as after:")
    print("  (rounds with an empty tentative pool are excluded: sum_p = 0")
    print("  makes their gamma a structural 0 where the budget cannot bind)")
    print(f"    {'arm':<22} {'gamma pre':>10} {'max':>6} {'post':>6} "
          f"{'over cap':>14} {'demoted/seed':>13}")
    print("    " + "-" * 76)
    gamma_binds = False
    for name, runs in arms.items():
        gs = [g for g in (replay_schedule(r) for r in runs.values()) if g]
        if not gs:
            continue
        n = len(gs)
        pre = statistics.mean(g["gamma_pre"] for g in gs)
        over = sum(g["rounds_over_budget"] for g in gs)
        live = sum(g["rounds"] - g["rounds_empty_pool"] for g in gs)
        print(f"    {name:<22} {pre:>10.3f} "
              f"{max(g['gamma_pre_max'] for g in gs):>6.3f} "
              f"{statistics.mean(g['gamma_post'] for g in gs):>6.3f} "
              f"{over}/{live} ({over / live:.0%})".ljust(0)
              + f"{sum(g['demoted_by_budget'] for g in gs) / n:>10.1f}")
        if pre > budget:
            gamma_binds = True
    print(f"    -> the budget {'DOES bind' if gamma_binds else 'does NOT bind'}")
    print()
    if sk40 >= 0.50:
        print("  F1 FIRES: the caps WERE the limiter.")
    elif abs(sk40 - sk_ref) < 0.05 and not gamma_binds:
        print("  F2 FIRES, both clauses: skip rate unchanged AND gamma slack.")
    elif abs(sk40 - sk_ref) < 0.05:
        print("  NEITHER falsifier fires as written. F1 needed a skip rate")
        print("  >= 50%; F2 needed gamma NOT binding. What happened instead:")
        print("  the caps are not the limiter AND the weight budget is -- the")
        print("  prediction's second ceiling turns out to be the first one.")
    else:
        print("  PARTIAL: the caps matter but not for the whole gap.")

    section("TEST 3  what actually decides each client-round?")
    print("  The scheduler's own Phase-1 branch order. Tier 2 is the final")
    print("  `else` and it PROMOTES, so a client under theta_high or with a")
    print("  short streak is skipped anyway -- it is NOT blocked. Then Phase 2")
    print("  demotes from the tentative pool until gamma <= gamma_budget.")
    _g0 = [g for g in (replay_schedule(r)
                       for r in arms[NO_EXPIRY].values()) if g]
    if _g0:
        print(f"  Per seed, out of "
              f"{sum(g['sampled'] for g in _g0) / len(_g0):.0f} sampled "
              f"client-rounds over "
              f"{sum(g['rounds'] for g in _g0) / len(_g0):.0f} rounds"
              + (f" ({sum(g['skipped_no_trust'] for g in _g0)} round(s) "
                 f"excluded for lack of a trust log -- TEST 1's rate spans "
                 f"ALL rounds, so the two differ)"
                 if sum(g["skipped_no_trust"] for g in _g0) else "."))
    print(f"\n  {'arm':<22} {'T3 prom':>8} {'T2 tent':>8} {'stale':>6} "
          f"{'tier1':>6} {'decoy':>6} {'demoted':>8} {'= skipped':>10}")
    print("  " + "-" * 80)
    for label, runs in [(REF_LABEL, ref_runs)] + list(arms.items()):
        gs = [g for g in (replay_schedule(r) for r in runs.values()) if g]
        if not gs:
            print(f"  {label:<22} -- no round-keyed trust log "
                  f"(predates the logging fix)")
            continue
        n = len(gs)
        print(f"  {label:<22} "
              + " ".join(f"{sum(g[k] for g in gs) / n:>{w}.1f}"
                         for k, w in (("tier3_promoted", 8),
                                      ("tier2_tentative", 8), ("stale", 6),
                                      ("tier1", 6), ("decoy", 6),
                                      ("demoted_by_budget", 8),
                                      ("replay_promoted", 10))))
    print("\n  Replay fidelity (counts are what the table reports):")
    for label, runs in arms.items():
        gs = [g for g in (replay_schedule(r) for r in runs.values()) if g]
        if not gs:
            continue
        n = len(gs)
        rp = sum(g["replay_promoted"] for g in gs) / n
        lp = sum(g["logged_promoted"] for g in gs) / n
        rounds = sum(g["rounds"] for g in gs)
        print(f"    {label:<22} replayed {rp:.1f} vs logged {lp:.1f} promoted "
              f"({abs(rp - lp) / lp:.1%} off); exact set match in "
              f"{sum(g['set_match'] for g in gs)}/{rounds} rounds")
    print("\n  The streak gate still matters, but for TIER, not for skipping:")
    for label, runs in arms.items():
        gs = [g for g in (replay_schedule(r) for r in runs.values()) if g]
        v = sum(g["verifications"] for g in gs)
        if not v:
            continue
        print(f"    {label:<22} streak reset on "
              f"{sum(g['streak_reset'] for g in gs) / v:>5.1%} of "
              f"verifications "
              f"(score<=0.8 {sum(g['reset_by_score'] for g in gs) / v:.1%}, "
              f"detector {sum(g['reset_by_detector'] for g in gs) / v:.1%})"
              f"; left at streak>=k_trust {sum(g['left_at_k'] for g in gs) / v:.1%}")
    tc = tavs_config(next(iter(arms[NO_EXPIRY].values())))
    lo, hi = (tc["pretrain_loss_var_rescale_lo"],
              tc["pretrain_loss_var_rescale_hi"])
    print(f"    a score > {SCORE_STREAK_CUTOFF} needs pretrain_loss_var < "
          f"{lo + (1 - SCORE_STREAK_CUTOFF) * (hi - lo):.3f}  "
          f"(window {lo}-{hi})")
    print("\n  A short streak demotes a client from Tier 3 to Tier 2. Tier 2")
    print("  still promotes, so the cost is losing the decoy exemption and")
    print("  being first in line for budget demotion -- not being blocked.")
    print("\n  Read with care: trust and the streak only update on")
    print("  verification, so a client held below the bar is partly held")
    print("  there BY being skipped. This sizes the gates; it does not")
    print("  settle the direction of causation.")

    section("TEST 4  does any of it cost accuracy?")
    print(f"  reference {REF.name}   round-to-round jitter {jitter(REF):.4f}")
    line, rule = header()
    print("  " + line)
    print("  " + rule)
    for name, path in ARMS:
        print("  " + fmt_row(name, ref_seeds, load_seeds(path)))
    print("\n  CI and batch-split check, last10 endpoint:")
    for name, path in ARMS:
        arm_seeds = load_seeds(path)
        shared = sorted(set(ref_seeds) & set(arm_seeds), key=int)
        diffs = [arm_seeds[s]["last10"] - ref_seeds[s]["last10"]
                 for s in shared]
        low, high, verdict = batch_agreement(ref_seeds, arm_seeds, "last10")
        print(f"    {name:<24} d={statistics.mean(diffs):+.4f} "
              f"+-{ci_halfwidth(diffs):.4f}   "
              f"seeds1-5 {low:+.4f} / 6-10 {high:+.4f}  agree={verdict}")

    section("TEST 5  WHY the null -- the budget absorbs what the cap releases")
    print("  If the budget is the binding constraint, relaxing the staleness")
    print("  cap cannot raise the skip rate: clients the cap stops forcing")
    print("  into V simply enter the tentative pool, and the demotion loop")
    print("  sends the surplus straight back. That predicts a specific")
    print("  bookkeeping identity -- stale falls, tentative promotions and")
    print("  budget demotions rise by about the same amount, and the number")
    print("  actually promoted does not move.")
    print(f"\n  {'arm':<22} {'stale':>7} {'tentative':>10} "
          f"{'demoted':>8} {'promoted':>9}")
    print("  " + "-" * 60)
    rows = []
    for label, runs in arms.items():
        gs = [g for g in (replay_schedule(r) for r in runs.values()) if g]
        if not gs:
            continue
        n = len(gs)
        row = {k: sum(g[k] for g in gs) / n
               for k in ("stale", "tier2_tentative", "tier3_promoted",
                         "demoted_by_budget", "replay_promoted")}
        row["tentative"] = row["tier2_tentative"] + row["tier3_promoted"]
        rows.append((label, row))
        print(f"  {label:<22} {row['stale']:>7.1f} {row['tentative']:>10.1f} "
              f"{row['demoted_by_budget']:>8.1f} "
              f"{row['replay_promoted']:>9.1f}")
    if len(rows) >= 2:
        (_, a), (_, b) = rows[0], rows[-1]
        d_stale = b["stale"] - a["stale"]
        d_tent = b["tentative"] - a["tentative"]
        d_dem = b["demoted_by_budget"] - a["demoted_by_budget"]
        d_prom = b["replay_promoted"] - a["replay_promoted"]
        print(f"\n  first arm -> last arm:  stale {d_stale:+.1f}, "
              f"tentative {d_tent:+.1f}, demoted {d_dem:+.1f}, "
              f"promoted {d_prom:+.1f}")
        print(f"  the cap released {-d_stale:.1f} client-rounds; the budget")
        print(f"  demoted {d_dem:+.1f} more of them, leaving promotion "
              f"{d_prom:+.1f}.")
        print("  That is the whole explanation of TEST 1's null, and it is a")
        print("  structural count rather than a noisy accuracy difference.")
    section("CAVEAT")
    print("  byzantine_fraction = 0 in every run here. The staleness cap is")
    print("  an anti-drift / anti-Sybil guard. Costing nothing WITHOUT an")
    print("  attacker says nothing about its value WITH one. Do not report a")
    print("  relaxed cap as safe on the strength of this sweep.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
