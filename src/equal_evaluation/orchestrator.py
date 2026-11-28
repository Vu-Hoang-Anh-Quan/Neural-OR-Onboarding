"""Orchestrator: POMO vs PyVRP vs OR-Tools on CVRP-50 (1000) + the 10 CVRPLIB-X
instances, same solver instances, shared time budgets.

Spec: `markdown/equal_evaluation_instructions.md` (sections 1-16 + appendices).

Spec run (NO CLI arguments; run from repo root; `SMOKE` below selects the gate):
    conda activate nor && python src/equal_evaluation/orchestrator.py

Phases
------
0. GPU gate: full `nvidia-smi` printed + recorded to `gpu_sm_query.txt`; then
   the free-VRAM decision (spec section 5):
     - SMOKE: free < ~1 GiB -> note + skip the POMO (GPU) part; smoke
       continues and still exits 0.
     - FULL: no CUDA / free < ~1 GiB -> stop (exit 2); free < ~2 GiB -> ask
       the user (interactive prompt; non-interactive = stop, exit 3).
    The free-VRAM check is re-run right before the decision (occupying
    processes change between snapshots).
1. Datagen (CPU): `datagen.generate_cvrp50` (always regenerates all N from
   scratch; deterministic seeds 1234+i, so a smoke batch of 2 is simply
   overwritten by the full batch of 1000 and the smoke POMO rows stay valid)
   + `datagen.load_x_set` (reads `data/chosen/*.vrp` + `.sol`, cross-checks
   against X_EXPECTED) + copy the X `.vrp` files to `results/equal_evaluation/x/`
   (datagen does NOT copy them; the solver CLIs expect `<x>/<name>.vrp`).
2. Reference (PyVRP, 60 s, seed 1, ONE committed row per instance): the single
   biggest wall cost (~16.7 h). FULL run: background subprocess
   `solvers_cli.py --reference` (resume-aware: committed rows are skipped, an
   already-running reference is adopted instead of double-starting), waited on
   before aggregate. SMOKE: sync `run_reference` on cvrp50 #0 only (~60 s).
3. POMO (GPU): `pomo_run.ensure_pomo` — per-instance `evaluate_policy` at all 3
   budgets (greedy / augment_dihedral_8 / sampling(1280)), 3 seed passes, every
   tour independently rechecked (coverage, route load <= Q, |reward + L| < tol)
   before its row commits -> `pomo_cvrp50.csv` / `pomo_x.csv`. FULL run then
   `pomo_run.ensure_budgets` -> `budgets.csv` (3 measured medians, merged into
   1 s / 10 s when within ~1%).
4. Budget solves (CPU, serial): both solver CLIs x every budget in
   `budgets.csv` (3 measured + fixed 1 s / 10 s) x seeds (FULL: 1/2/3; SMOKE: 1)
   -> `solver_budgets_pyvrp.csv` (one row per file,budget,seed) and
   `solver_budgets_ortools.csv` (one row per file,budget; OR-Tools has no seed
   arg, so up to 3 identical re-runs until a row commits — sub-1 s budgets die
   with rc=1 and may commit NO row: documented, the cell stays empty).
   FAILURES (rc != 0 / unparseable / infeasible) are logged with NO committed
   row -> possibly-empty table cells (documented in results.md).
5. Report (FULL only): wait for the background reference, then
   `aggregate.main(RES_DIR)` -> `results.md` + the two gap-vs-time PNGs.
   SMOKE instead runs the smoke gate (assertions below) and exits 0.

Smoke gate (SMOKE = True; must exit 0 end-to-end, <= ~5 min)
------------------------------------------------------------
- instances: CVRP-50 #0 and #1 + X-n502-k39; reference 60 s on #0 (sync);
  POMO all 3 budgets on all 3 instances; budget solves [1 s, 10 s], seed 1,
  both solvents via their CLIs.
- asserts (each fails the smoke with a clear message):
  (i)  POMO committed rows: pomo_cvrp50.csv has pass-0 rows for ids 0 and 1 in
        all 3 modes (rows only commit after the independent recheck passed);
  (ii) reference row for cvrp50_0000 present with numeric cost > 0 and
        wall > 0 (the 60 s solve committed a feasible, parsed row);
  (iii) budget rows: solver_budgets_pyvrp.csv has >= 1 committed data row;
  (iv)  live n502 pass-0 (the spec's sanity anchor, pomo_x.csv id =
        "X-n502-k39", pass 0): L = -reward x 1000 within 2 % of
        greedy 249374 / augment_dihedral_8 96986 / sampling 189837 (BKS 69226,
        expected gaps +260 % / +40 % / +174 %), all feasible (committed rows
        are rechecked rows).
- GPU part skipped (free < ~1 GiB): the POMO-dependent asserts (i) and (iv) are
  skipped with a note; the smoke still verifies phases 1/2/4 and exits 0.

Re-running is resumable: every CSV is append-only keyed; committed keys are
skipped; a crashed run is restarted and picks up where it left off.
"""

import csv
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # src/equal_evaluation
SRC_DIR = THIS_DIR.parent                            # src/ (utils/ lives here)
ROOT = SRC_DIR.parent                                # repo root
sys.path.insert(0, str(SRC_DIR))                    # house pattern: src/ on sys.path

from utils.logging_utils import setup_logging  # noqa: E402

import datagen  # noqa: E402
import pomo_run  # noqa: E402  (module-level torch/rl4co import, no model load yet)
import solvers_cli  # noqa: E402
import aggregate  # noqa: E402

# ---------------------------------------------------------------------------
# Switch + constants
# ---------------------------------------------------------------------------

SMOKE = False  # True = smoke gate (spec section 13/15); False = full run.

N_CVRP_FULL = 1000              # full-run CVRP-50 count
SMOKE_CVRP_IDS = (0, 1)         # smoke covers #0 and #1
SMOKE_X_NAME = "X-n502-k39"     # the spec's X sanity instance (n=502, BKS 69226)
BUDGET_SEEDS = (1, 2, 3)        # PyVRP budget-seed set (full run)
SMOKE_BUDGET_SEEDS = (1,)

# GPU gate free-VRAM thresholds (MiB).
GPU_STOP_FREE_MIB = 1024        # < ~1 GiB: skip (smoke) / stop (full)
GPU_ASK_FREE_MIB = 2048         # < ~2 GiB (full only): ask the user

RES_DIR = ROOT / "results" / "equal_evaluation"

# Smoke gate: n502 pass-0 anchor values (spec appendix; L = -reward x 1000,
# coords [0,1000]/1000 = [0,1] scale). 2 % relative tolerance: the policy +
# recheck recipe are fixed and the probe reproduced these to the digit, but
# CUDA run-to-run atomics leave a tiny margin.
SMOKE_N502_L = {
    "greedy": 249_374,
    "augment_dihedral_8": 96_986,
    "sampling": 189_837,
}
SMOKE_N502_BKS = 69_226
SMOKE_N502_TOL = 0.02

# Background reference bookkeeping.
REF_CSV = RES_DIR / "reference_pyvrp_60s.csv"
REF_LOG = RES_DIR / "reference_ref.log"
REF_PID = RES_DIR / "reference_ref.pid"


# ---------------------------------------------------------------------------
# 0. GPU gate
# ---------------------------------------------------------------------------

def _smi_full_text() -> str:
    try:
        q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60)
        if q.returncode != 0:
            return f"(nvidia-smi exited rc={q.returncode})\n{q.stderr[-400:]}"
        return q.stdout
    except Exception as e:  # FileNotFoundError / timeout / OSError
        return f"(nvidia-smi failed: {e})"


def _smi_query_line(log: logging.Logger) -> str:
    """The spec's one-line query (recorded at log start + in results.md
    § Hardware via gpu_sm_query.txt, which aggregate reads and inlines)."""
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        if q.returncode != 0:
            return f"(nvidia-smi query failed: rc={q.returncode})"
        line = q.stdout.strip()
        log.info("gpu query line: %s", line)
        return line
    except Exception as e:
        return f"(nvidia-smi query failed: {e})"


def _smi_free_mi_b() -> list[int]:
    """Free VRAM (MiB) per GPU via the CLI; [] when unusable."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return []
    if q.returncode != 0:
        return []
    out = []
    for line in q.stdout.strip().splitlines():
        s = line.strip()
        if s.lstrip("-").isdigit():
            out.append(int(s))
    return out


def record_pre_run_gpu(log: logging.Logger) -> int:
    """Spec 'Mandatory pre-run gate' (record half): print full `nvidia-smi`,
    query + log the one-line `--query-gpu=name,memory.free` line, and record
    it in `gpu_sm_query.txt` (aggregate inlines it into results.md § Hardware).
    Returns the min free MiB (0 when CUDA/nvidia-smi unusable)."""
    print("\n================ nvidia-smi (full, pre-run gate) ================")
    print(_smi_full_text())
    (RES_DIR / "gpu_sm_query.txt").write_text(_smi_query_line(log) + "\n")
    free = _smi_free_mi_b()
    free_min = min(free) if free else 0
    log.info("pre-run GPU gate: full nvidia-smi printed; free VRAM %.2f GiB",
             free_min / 1024)
    return free_min


def decide_gpu_gate(log: logging.Logger, smoke: bool) -> bool:
    """Decide whether the POMO (GPU) part runs; re-checks free VRAM at this
    instant (occupying processes change between snapshots — this runs right
    before the GPU part).

    SMOKE: free < ~1 GiB -> note + return False (smoke continues, exit 0).
    FULL (spec pre-run gate): no CUDA / free < ~1 GiB -> stop (exit 2);
    free < ~2 GiB -> ask (exit 3 when declined or non-interactive).
    Returns True to run the GPU part.
    """
    free = _smi_free_mi_b()
    if not free:
        log.warning("GPU gate: nvidia-smi reports no usable CUDA")
        if smoke:
            log.warning("GPU gate (smoke): skipping the POMO (GPU) part; "
                        "smoke continues and exits 0")
            return False
        print("GPU gate: no usable CUDA. FULL run stopped (exit 2): get a "
              "CUDA device / free VRAM and re-run (spec pre-run gate).")
        sys.exit(2)

    free_min = min(free)
    print(f"-- GPU gate (re-checked now): free VRAM {free_min / 1024:.2f} GiB "
          f"({len(free)} GPU(s) visible)")

    if free_min < GPU_STOP_FREE_MIB:
        if smoke:
            log.warning("GPU gate (smoke): only %.2f GiB free (< ~1 GiB) — "
                        "skipping the POMO (GPU) part; smoke continues",
                        free_min / 1024)
            return False
        print(f"GPU gate: only {free_min / 1024:.2f} GiB free (< ~1 GiB). "
              "FULL run stopped (exit 2): re-run when the GPU has "
              "~1-2 GiB free (spec pre-run gate).")
        sys.exit(2)
    if not smoke and free_min < GPU_ASK_FREE_MIB:
        print(f"GPU gate: {free_min / 1024:.2f} GiB free (< ~2 GiB, full run). "
              "The POMO phase (batch-1) will run; OOM/slowdown risk is higher "
              "than the proven ~1-2 GiB headroom. Spec says to ask the user.")
        if sys.stdin.isatty():
            ans = input("Continue with the POMO phase anyway? [y/N] ").strip().lower()
        else:
            print("non-interactive run: cannot ask; treating as 'no'.")
            ans = ""
        if ans not in ("y", "yes"):
            sys.exit(3)
    return True


# ---------------------------------------------------------------------------
# 1. Datagen
# ---------------------------------------------------------------------------

def phase_datagen(n_cvrp: int, log: logging.Logger):
    """CVRP-50 .vrp + generator tds (always regenerated, idempotent) and the X
    record list. Returns (gen dict, X recs)."""
    gen = datagen.generate_cvrp50(n_cvrp, RES_DIR, log)
    recs = datagen.load_x_set(RES_DIR, ROOT / "data" / "chosen", log)
    return gen, recs


def copy_x_vrp(log: logging.Logger, x_names: list[str]) -> None:
    """Copy the (cross-checked) X .vrp files next to their names into
    RES_DIR/x/, where the solver CLIs expect them (`<x>/<name>.vrp`)."""
    xdir = RES_DIR / "x"
    xdir.mkdir(parents=True, exist_ok=True)
    for name in x_names:
        src = ROOT / "data" / "chosen" / f"{name}.vrp"
        dst = xdir / f"{name}.vrp"
        shutil.copy2(src, dst)
        log.info("copied X instance file %s -> %s", src.name, dst)


# ---------------------------------------------------------------------------
# 2. Reference
# ---------------------------------------------------------------------------

def _reference_running() -> list[int]:
    """pgrep for an existing reference subprocess; [] when none."""
    try:
        q = subprocess.run(["pgrep", "-f", "solvers_cli.py --reference"],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if q.returncode != 0:  # pgrep: no match (or missing; both -> none running)
        return []
    return [int(p) for p in q.stdout.split() if p.strip().isdigit()]


def start_reference_bg(log: logging.Logger):
    """Start (or adopt) the background reference subprocess and return the
    Popen handle (None when adopting an externally-running one)."""
    existing = _reference_running()
    if existing:
        log.warning("reference subprocess already running (pids %s); adopting "
                    "it, will wait on it before aggregate", existing)
        REF_PID.unlink(missing_ok=True)
        return None
    logf = open(REF_LOG, "a")
    logf.write(f"\n=== solvers_cli.py --reference started "
               f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    logf.flush()
    proc = subprocess.Popen(
        [sys.executable, str(THIS_DIR / "solvers_cli.py"), "--reference"],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
    )
    REF_PID.write_text(str(proc.pid))
    log.info("background reference subprocess started (pid %s, log %s)",
             proc.pid, REF_LOG.name)
    return proc


def wait_reference(log: logging.Logger, proc) -> None:
    """Block until the background reference is done; then sanity-check the
    reference CSV (aggregate will SystemExit with a clear message if the CSV
    is missing — designed behavior; missing rows just stay missing)."""
    log.info("waiting for the background reference to finish ...")
    if proc is not None:
        rc = proc.wait()
    else:
        # Adopted external process: poll pgrep until it's gone.
        while _reference_running():
            time.sleep(30)
        rc = None
    REF_PID.unlink(missing_ok=True)
    if proc is not None and rc != 0:
        log.warning("background reference exited rc=%s — if rows are "
                    "missing, re-run the orchestrator to resume", rc)
    if not REF_CSV.exists():
        log.error("reference CSV missing after wait (see %s); aggregate will "
                  "refuse to run — fix the reference run and re-run", REF_LOG.name)
    else:
        n = sum(1 for _ in open(REF_CSV)) - 1
        log.info("reference done: %d rows committed", n)


# ---------------------------------------------------------------------------
# Smoke gate (SMOKE only)
# ---------------------------------------------------------------------------

def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def smoke_gate(log: logging.Logger, gpu_ok: bool) -> None:
    """Assert the smoke ran as specified; any AssertionError fails the smoke."""
    cvrp_names = [f"cvrp50_{i:04d}" for i in SMOKE_CVRP_IDS]
    p50 = RES_DIR / "pomo_cvrp50.csv"
    px = RES_DIR / "pomo_x.csv"
    ref = RES_DIR / "reference_pyvrp_60s.csv"
    bpy = RES_DIR / "solver_budgets_pyvrp.csv"
    bo = RES_DIR / "solver_budgets_ortools.csv"

    rows50 = _read_rows(p50)
    rowsx = _read_rows(px)
    rowsref = _read_rows(ref)
    rowspy = _read_rows(bpy)
    rowsort = _read_rows(bo)

    # (i) POMO committed (post-recheck) rows, pass 0, all 3 modes, both sets.
    if gpu_ok:
        got50 = {(r["id"], r["pass"], r["mode"]) for r in rows50}
        # CSV ids are strings (pomo_run writes str(i)); compare as strings.
        need50 = {(str(i), "0", m) for i in SMOKE_CVRP_IDS for m in pomo_run.MODES}
        missing = need50 - got50
        assert not missing, f"pomo_cvrp50.csv missing pass-0 rows: {sorted(missing)}"
        gotx = {(r["id"], r["pass"], r["mode"]) for r in rowsx}
        needx = {(SMOKE_X_NAME, "0", m) for m in pomo_run.MODES}
        missingx = needx - gotx
        assert not missingx, f"pomo_x.csv missing n502 pass-0 rows: {sorted(missingx)}"
        # (iv) live n502 pass-0 anchor values.
        L = {r["mode"]: -float(r["reward"]) * 1000.0 for r in rowsx
             if r["id"] == SMOKE_X_NAME and r["pass"] == "0"}
        for mode, expect in SMOKE_N502_L.items():
            assert mode in L, f"n502 pass-0 {mode} row missing"
            assert abs(L[mode] - expect) / expect <= SMOKE_N502_TOL, (
                f"n502 pass-0 {mode}: L={L[mode]:.0f} not within "
                f"{SMOKE_N502_TOL:.0%} of expected {expect}")
            assert L[mode] > SMOKE_N502_BKS, (
                f"n502 pass-0 {mode}: L={L[mode]:.0f} <= BKS {SMOKE_N502_BKS} "
                f"(POMO is expected to be WORSE than BKS; gap should be +)")
            log.info("smoke n502 pass-0 %s: L=%.0f (expected ~%s, BKS %s)",
                     mode, L[mode], f"{expect:,}", f"{SMOKE_N502_BKS:,}")
    else:
        log.warning("smoke gate: GPU part skipped — asserts (i)/(iv) skipped")

    # (ii) reference: a committed, numeric, >0 row for cvrp50 #0 (sync 60 s).
    refrow = [r for r in rowsref if r["file"] == f"cvrp50_{SMOKE_CVRP_IDS[0]:04d}"]
    assert refrow, "reference_pyvrp_60s.csv: no row for cvrp50 #0 (the 60 s " \
                  "solve must commit a parsed, feasible row)"
    cost = float(refrow[0]["cost"])
    assert cost > 0 and float(refrow[0]["wall"]) > 0, \
        f"reference row not sane: {refrow[0]}"
    log.info("smoke reference cvrp50 #0: cost=%s wall=%s (x1e6 scale)",
             cost, refrow[0]["wall"])

    # (iii) budget rows committed (PyVRP at least — OR-Tools sub-1 s may
    # legitimately commit nothing; its row presence is not asserted).
    assert rowspy, "solver_budgets_pyvrp.csv: no committed rows (PyVRP must " \
                  "commit at least its 10 s rows in the smoke)"
    log.info("smoke budgets: pyvrp=%d rows, ortools=%d rows committed",
             len(rowspy), len(rowsort))

    # Note (not an assert): all POMO rows that got here were rechecked in
    # pomo_run before commit (coverage / load<=Q / |reward+L|<tol) — (i) and
    # (iv) covering their presence covers the recheck requirement.
    log.info("smoke gate: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(RES_DIR)
    log = logging.getLogger("equal_evaluation.orchestrator")

    smoke = SMOKE
    n_cvrp = 2 if smoke else N_CVRP_FULL
    x_names = [SMOKE_X_NAME] if smoke else None  # None -> all 10 from x_records
    cvrp_names = [f"cvrp50_{i:04d}" for i in SMOKE_CVRP_IDS]

    log.info("=== equal_evaluation orchestrator (smoke=%s, n_cvrp=%d) ===",
             smoke, n_cvrp)

    # ---- 0. Mandatory pre-run gate (record half): print full `nvidia-smi`,
    #      log + record the one-line query in gpu_sm_query.txt (aggregate
    #      inlines it into results.md § Hardware); fail fast on NO CUDA at
    #      all (the spec requires CUDA available: True pre-run — the
    #      free-VRAM < ~1 GiB / < ~2 GiB decisions happen right before the
    #      GPU part, phase 3, with a fresh re-check at that instant).
    pre_run_free = record_pre_run_gpu(log)
    if pre_run_free == 0 and not smoke:
        print("Pre-run gate: no usable CUDA (free-VRAM query empty). FULL run "
              "stopped (exit 2).")
        sys.exit(2)

    # ---- 1. Datagen
    log.info("=== phase 1: datagen (CVRP-50 x%d + the 10 X instances) ===", n_cvrp)
    gen, recs = phase_datagen(n_cvrp, log)
    if x_names is None:
        x_names = [r["name"] for r in recs]
    copy_x_vrp(log, x_names)

    # ---- 2. Reference (background for the full run; sync for the smoke)
    log.info("=== phase 2: reference (PyVRP 60 s, seed 1) ===")
    ref_proc = None
    if smoke:
        log.info("smoke: sync reference 60 s on cvrp50 #0 (~60 s)")
        solvers_cli.run_reference(RES_DIR, [cvrp_names[0]], log)
    else:
        ref_proc = start_reference_bg(log)

    # ---- 3. POMO (GPU): the free-VRAM gate decision, with a fresh re-check
    #      of free VRAM right at this instant (occupying processes change
    #      between snapshots).
    log.info("=== phase 3: POMO (GPU) ===")
    gpu_ok = decide_gpu_gate(log, smoke)
    if gpu_ok:
        pomo_run.ensure_pomo(RES_DIR, n_cvrp, x_names, log)
        if not smoke:
            pomo_run.ensure_budgets(RES_DIR, log)
    else:
        log.warning("POMO GPU part skipped (see gate above) — pomo_cvrp50.csv / "
                    "pomo_x.csv rows will be absent and their table cells empty")

    # ---- 4. Budget solves
    log.info("=== phase 4: budget solves (both solver CLIs, serial) ===")
    if smoke:
        all_budgets = [1.0, 10.0]
        b_seeds = SMOKE_BUDGET_SEEDS
        cvrp50_for_budgets = cvrp_names
    else:
        bpath = RES_DIR / "budgets.csv"
        if bpath.exists():
            with open(bpath) as f:
                measured = [float(r["column_time_s"]) for r in csv.DictReader(f)]
        else:
            measured = []  # GPU part skipped -> fixed budgets only (documented)
        all_budgets = sorted(set(measured) | set(pomo_run.FIXED_BUDGETS))
        b_seeds = BUDGET_SEEDS
        # run_budgets takes explicit names (None = empty); derive the full
        # CVRP-50 name list from the generated .vrp files (same recipe as
        # solvers_cli's own __main__).
        cvrp50_for_budgets = solvers_cli._cvrp50_names()
    log.info("budgets: %s | pyvrp seeds: %s | cvrp50: %d | x: %d",
             all_budgets, b_seeds, len(cvrp50_for_budgets), len(x_names))
    solvers_cli.run_budgets(RES_DIR, all_budgets, log,
                            cvrp50_names=cvrp50_for_budgets,
                            x_names=x_names, seeds=b_seeds)

    # ---- 5. Report (full) / smoke gate
    if not smoke:
        wait_reference(log, ref_proc)
        aggregate.main(RES_DIR)
    else:
        smoke_gate(log, gpu_ok)

    log.info("=== orchestrator complete (smoke=%s) ===", smoke)


if __name__ == "__main__":
    main()
