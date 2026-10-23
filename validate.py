"""
Validation suite for the news-filtered intraday reversal strategy.

    python validate.py <subcommand> [--synthetic] [--workers N] [--unlock-holdout] [overrides]

Subcommands:
    coverage      data audit (news coverage per year, missing bars)
    correctness   planted/null tests, lookahead traps, delay test, trade audit, pytest
    stats         group stats, block bootstrap, random-entry baseline, placebo news, deflated Sharpe
    permutation   Monte Carlo permutation test of the full pipeline incl. the in-sample sweep
    synthetic     controlled-truth markets (GARCH-t generator fitted to the data)
    robustness    walk-forward, heatmaps, jitter, breakdowns, concentration, out-of-universe
    execution     cost multiples, pessimistic fills, missed fills, delays, short-borrow stress
    risk          Monte Carlo drawdowns, position sizing, kill criteria
    all           everything above, then REPORT.md
    report        rebuild REPORT.md from saved results

Outputs: results/validation/<subcommand>/ (real data) or results/validation_synthetic/<subcommand>/.
The holdout (last 12 months) is never loaded unless --unlock-holdout is given; each unlock is logged.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

import config as C

SUBCOMMANDS = ["coverage", "correctness", "stats", "permutation", "synthetic", "robustness", "execution", "risk"]
# `all` runs stats after the sweeps so the Deflated Sharpe sees every logged trial.
ALL_ORDER = ["coverage", "correctness", "permutation", "robustness", "synthetic", "stats", "execution", "risk"]
MODULES = {"synthetic": "scenarios"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=SUBCOMMANDS + ["all", "report"])
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic base data (no API keys needed)")
    ap.add_argument("--workers", type=int, default=C.N_WORKERS)
    ap.add_argument("--heavy-workers", type=int, default=C.N_WORKERS_HEAVY, help="workers for the permutation test")
    ap.add_argument("--unlock-holdout", action="store_true", help="include the locked last 12 months (logged!)")
    ap.add_argument("--permutations", type=int, default=None, help=f"default {C.PERMUTATION_N}")
    ap.add_argument("--scenario-runs", type=int, default=None, help=f"default {C.SCENARIO_RUNS}")
    ap.add_argument("--bootstrap", type=int, default=None, help=f"default {C.BOOTSTRAP_N}")
    ap.add_argument("--jitter", type=int, default=None, help=f"default {C.JITTER_N}")
    args = ap.parse_args()

    from validation import common, final_report
    opts = {"permutations": args.permutations, "scenario_runs": args.scenario_runs,
            "bootstrap": args.bootstrap, "jitter": args.jitter, "heavy_workers": args.heavy_workers}
    if args.command == "report":
        out_root = C.VALIDATION_SYNTH_DIR if args.synthetic else C.VALIDATION_DIR
        final_report.write(out_root, args.synthetic)
        return

    t0 = time.time()
    ctx = common.load_context(args.synthetic, args.unlock_holdout, args.workers, opts, " ".join(sys.argv))
    todo = ALL_ORDER if args.command == "all" else [args.command]
    for name in todo:
        t1 = time.time()
        print(f"\n===== {name} =====", flush=True)
        mod = importlib.import_module(f"validation.{MODULES.get(name, name)}")
        mod.run(ctx)
        print(f"===== {name} done [{time.time() - t1:.0f}s] =====", flush=True)
    if args.command == "all":
        final_report.write(ctx.out_root, args.synthetic)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
