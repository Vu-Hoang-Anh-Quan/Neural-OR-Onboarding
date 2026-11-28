"""Solver-CLI phase for the equal-budget evaluation (no re-implementation).

Per instance, per time limit, per seed: `subprocess.run` the existing
`src/run_pyvrp.py` / `src/run_ortools.py` CLIs (cwd = repo root, so their
`from cvrp.cvrp_utils import ...` resolves), parse their one-line CSV stdout:
  pyvrp,{path},{sec},{seed},{cost},{is_feasible},{wall}
  ortools,{path},{sec},-,{cost},{check_ok},{wall}
Failures (rc != 0, no/uncsvable line, feasible == False) are logged and the
key stays uncommitted — the cell may be empty in the table (OR-Tools dies with
rc=1 at sub-second budgets: live-verified `AttributeError` at
`run_ortools.py` line 26).

Reference (CVRP-50 set only): `run_pyvrp.py <file> 60 1` per instance,
single fixed seed = the per-instance reference target (NOT the randomized
object; 3 seeds would triple ~16.7 h of serial solves with no change to the
reference semantics). Write the row per instance as it completes — a crashed
run restarts only the missing ones.

CSVs (append-only; one committed row per key; resume = skip committed keys):
  <results>/solver_budgets_pyvrp.csv   file, budget, seed, cost, wall
  <results>/solver_budgets_ortools.csv file, budget, cost, wall    (no seed arg on the CLI)
  <results>/reference_pyvrp_60s.csv    file, cost, wall
`budget` column format: `f"{b:g}"` (1.5 / 2.4 / 9.4 / 1 / 10), shared with
the measured column decisions in budgets.csv and the aggregate.
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "results" / "equal_evaluation"
PYVRP_CLI = ROOT / "src" / "run_pyvrp.py"
ORTOOLS_CLI = ROOT / "src" / "run_ortools.py"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logging_utils import setup_logging  # noqa: E402

# The reference: PyVRP 60 s, single fixed seed (see module docstring).
REF_SECONDS = 60
REF_SEED = 1

# Budget-run wall timeouts (generous over the budget; the CLIs honor the
# MaxRuntime / time_limit internally, these are hard safety valves).
BUDGET_TIMEOUT_MARGIN_S = 30
BUDGET_TIMEOUT_MAX_S = 5


def budget_sec_str(b: float) -> str:
    """Canonical sec-string for a budget value (1.5 -> '1.5', 1.0 -> '1')."""
    return f"{b:g}"


def _parse_csv_line(stdout: str):
    """Last non-empty stdout line -> (cost: float, feasible: bool) or None.

    Both CLIs print exactly one CSV line (7 fields); treat any other output as
    a failure (no line / wrong shape / cost column not numeric)."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    parts = lines[-1].split(",")
    if len(parts) != 7:
        return None
    try:
        cost = float(parts[4])
    except ValueError:
        return None
    return cost, parts[5].strip() == "True"


def _file_path(res_dir: Path, name: str) -> Path:
    """Instance file for a name: cvrp50 names live under res_dir/'cvrp50/',
    X names under res_dir/'x/' as <name>.vrp."""
    if name.startswith("cvrp50_"):
        return res_dir / "cvrp50" / f"{name}.vrp"
    return res_dir / "x" / f"{name}.vrp"


def _committed_keys(path: Path, keyfields: tuple[str, ...]) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        return {tuple(row[k] for k in keyfields)
                for row in csv.DictReader(f) if row.get(keyfields[0])}


def _append_csv(path: Path, header: list[str], row: dict):
    first = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first:
            w.writeheader()
        w.writerow(row)
        f.flush()


# --- the reference run (CVRP-50 set) --------------------------------------------

def run_reference(res_dir: Path, cvrp50_names: list, log) -> None:
    """run_pyvrp.py <file> 60 1 per instance; per-instance row file (resume).
    THE single biggest wall-clock cost (~16.7 h) — run first, in the
    background, if the user approves."""
    out = res_dir / "reference_pyvrp_60s.csv"
    keyfields = ("file",)
    committed = _committed_keys(out, keyfields)
    log.info("reference: %d instances, %d committed (resume)", len(cvrp50_names), len(committed))
    header = ["file", "cost", "wall"]
    for name in cvrp50_names:
        if (name,) in committed:
            continue
        file_path = _file_path(res_dir, name)
        res = _one_run("pyvrp", file_path, REF_SECONDS, seed=REF_SEED, log=log)
        if res is None:
            continue  # logged; key uncommitted -> retried next run
        cost, _, wall = res
        _append_csv(out, header, {"file": name, "cost": f"{cost:.4f}", "wall": f"{wall:.2f}"})
        committed.add((name,))
        n_committed = len(committed)
        if n_committed % 50 == 0:
            log.info("reference: %d/%d done", n_committed, len(cvrp50_names))
    log.info("reference: done (%d committed)", len(committed))


# --- budget runs (both sets, 5 columns) ------------------------------------------

def run_budgets(res_dir: Path, all_budgets: list, log,
                cvrp50_names: list | None = None, x_names: list | None = None,
                seeds: tuple = (1, 2, 3)) -> None:
    """Both CLIs x `all_budgets` (3 measured + 1 s + 10 s) x PyVRP seeds
    (full run: 1/2/3 per the spec; smoke gate: seed 1 only) x OR-Tools
    re-runs per budget (OR-Tools has no seed arg — up to 3 attempts until one
    committed row, keyed by (file, budget), so repeated cells pool as repeats)."""
    names = (cvrp50_names or []) + (x_names or [])
    out_py = res_dir / "solver_budgets_pyvrp.csv"
    out_ort = res_dir / "solver_budgets_ortools.csv"
    header_py = ["file", "budget", "seed", "cost", "wall"]
    header_ort = ["file", "budget", "cost", "wall"]
    comm_py = _committed_keys(out_py, ("file", "budget", "seed"))
    comm_ort = _committed_keys(out_ort, ("file", "budget"))
    log.info("budgets: %d instances x %d budgets x seeds %s (pyvrp) + %d x %d (ortools); "
             "committed pyvrp=%d ortools=%d",
             len(names), len(all_budgets), seeds, len(names), len(all_budgets),
             len(comm_py), len(comm_ort))
    for name in names:
        file_path = _file_path(res_dir, name)
        for b in all_budgets:
            b_str = budget_sec_str(b)
            # PyVRP: one row per seed (spec: 1/2/3 full run; 1 smoke)
            for seed in seeds:
                if (name, b_str, str(seed)) in comm_py:
                    continue
                res = _one_run("pyvrp", file_path, b, seed=seed, log=log)
                if res is None:
                    continue
                cost, _, wall = res
                _append_csv(out_py, header_py,
                           {"file": name, "budget": b_str, "seed": seed,
                            "cost": f"{cost:.4f}", "wall": f"{wall:.2f}"})
                comm_py.add((name, b_str, str(seed)))
            # OR-Tools: 3 identical re-runs keyed by (file, budget) — the cell
            # pools them as repeats (OR-Tools is deterministic given the same
            # seed; std ~= 0, documented in results.md).
            for rep in range(3):
                if (name, b_str) in comm_ort:
                    break
                res = _one_run("ortools", file_path, b, seed=None, log=log)
                if res is None:
                    continue
                cost, _, wall = res
                _append_csv(out_ort, header_ort,
                           {"file": name, "budget": b_str, "cost": f"{cost:.4f}", "wall": f"{wall:.2f}"})
                comm_ort.add((name, b_str))
            log.debug("budgets: %s done all budgets", name)
    log.info("budgets: done (pyvrp=%d ortools=%d)", len(comm_py), len(comm_ort))


def _one_run(solver: str, file_path: Path, sec: float, seed=None, log=None, timeout: int = None):
    """Run one CLI; return (cost, feasible, wall) or None on any failure."""
    if timeout is None:
        timeout = max(BUDGET_TIMEOUT_MAX_S, int(sec) + BUDGET_TIMEOUT_MARGIN_S)
    cli = PYVRP_CLI if solver == "pyvrp" else ORTOOLS_CLI
    args = [str(cli), str(file_path), budget_sec_str(sec)]
    if solver == "pyvrp":
        args.append(str(seed))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable] + args, capture_output=True, text=True,
                              cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        log.warning("budgets: TIMEOUT %s %s sec=%s (wall=%.1fs) — no row (retry next run)",
                    solver, file_path.name, budget_sec_str(sec), wall)
        return None
    wall = time.perf_counter() - t0
    parsed = _parse_csv_line(proc.stdout)
    label = f"{solver} {file_path.name} sec={budget_sec_str(sec)}"
    if proc.returncode != 0:
        # Live-verified OR-Tools failure mode: rc=1 at sub-second budgets
        # (`AttributeError` at `sol.Value(rm.NextVar(i))`, no CSV line).
        tail = proc.stderr.strip().splitlines()[-1:] or ["(empty stderr)"]
        log.warning("budgets: FAILED %s rc=%d (no CSV line; cell may be empty) stderr_tail=%r",
                    label, proc.returncode, [t for t in tail][:1])
        return None
    if parsed is None:
        out_tail = proc.stdout.strip().splitlines()[-1:] or ["(empty stdout)"]
        log.warning("budgets: UNPARSED %s stdout_tail=%r", label, [t for t in out_tail][:1])
        return None
    cost, feasible = parsed
    if not feasible:
        log.warning("budgets: INFEASIBLE %s cost=%s — logged, no row", label, cost)
        return None
    return cost, True, wall


def _cvrp50_names() -> list[str]:
    """The CVRP-50 instance names, derived from the generated .vrp files
    (the datagen phase must have run; the count is discovered, not assumed)."""
    vrp_dir = RES_DIR / "cvrp50"
    names = sorted(p.stem for p in vrp_dir.glob("cvrp50_*.vrp"))
    assert names, f"no generated CVRP-50 .vrp files found under {vrp_dir} — run the datagen phase first"
    return names


if __name__ == "__main__":
    # Background reference entry point (the orchestrator Popens this with cwd =
    # the repo root, so `from cvrp.cvrp_utils import ...` inside run_pyvrp.py
    # resolves as it does from the interactive shell).
    parser = argparse.ArgumentParser(description="equal_evaluation: PyVRP reference phase")
    parser.add_argument("--reference", action="store_true",
                        help="run the per-instance PyVRP 60 s seed=1 reference")
    args = parser.parse_args()
    assert args.reference, "--reference is the only supported mode"

    RES_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(RES_DIR)  # root logger -> results/equal_evaluation/log.txt + console
    log = logging.getLogger("equal_evaluation.reference")
    run_reference(RES_DIR, _cvrp50_names(), log)
