"""Aggregation for the equal-budget evaluation: pooled per-instance gap tables,
``results.md`` (house Environment/Hardware sections), and the two PNGs.

Reads only committed CSVs + ``budgets.csv`` (no datagen/pomo_run imports, so it
stays light; heavy deps are numpy + matplotlib only). Per instance i and pass p:
``gap% = 100 * (cost - ref) / ref`` (positive = worse). A cell is the mean +-
population std (ddof=0) of the **pooled per-instance gaps** — the (instance,
pass) pairs enter the pool, giving (1000 instances x 3 passes) for CVRP-50 and
(10 instances x 3 passes) for the X set.

References: X set -> BKS from ``x_records.csv``; CVRP-50 set -> per-instance
PyVRP-60s seed=1 cost from ``reference_pyvrp_60s.csv``. Costs: POMO
``-reward * 1e6`` (CVRP-50) / ``-reward * 1000`` (X); PyVRP printed float;
OR-Tools printed int (both already on the instance's coordinate scale).

Columns: the 3 measured POMO-mode times (from ``budgets.csv``), then fixed
``1 s``, ``10 s``. POMO fills only its measured columns (its budget is the
decode time, not a wall-clock limit; its 1 s / 10 s cells stay blank). PyVRP and
OR-Tools fill all 5.

Plots: x = budget seconds (log), 5 categorical ticks, 3 default-color lines
with a +-1 std ``fill_between`` band; POMO starts at its own first point and,
whenever all 3 measured times are < 10 s, a horizontal dotted line extends it
to the right edge (the caption notes POMO's budget is decode-time, not
wall-clock).
"""
from __future__ import annotations

import csv
import os
import platform
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODE_ORDER = ("greedy", "augment_dihedral_8", "sampling")
MODE_LABELS = {"greedy": "greedy", "augment_dihedral_8": "dihedral-8",
               "sampling": "sampling/1280"}
METHODS = ("POMO", "PyVRP", "OR-Tools")
METHOD_COLOR = {"POMO": "tab:blue", "PyVRP": "tab:orange", "OR-Tools": "tab:green"}
SOLVER_CSV = {"PyVRP": "solver_budgets_pyvrp.csv",
              "OR-Tools": "solver_budgets_ortools.csv"}


# ---------------------------------------------------------------------------
# CSV loading (fail loudly if a phase file is missing)

def _f(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"aggregate: missing {path} - the evaluation phases are "
                         "not complete (see log.txt)")
    return path


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_budgets(res: Path) -> tuple[dict, list]:
    """(mode -> column-seconds float, ordered mode list)."""
    rows = _read_csv(_f(res / "budgets.csv"))
    modes = [r["mode"] for r in rows]
    times = {r["mode"]: float(r["column_time_s"]) for r in rows}
    return times, modes


def load_reference(res: Path) -> dict:
    """CVRP-50 name -> reference cost (per-instance PyVRP 60 s, seed=1)."""
    return {r["file"]: float(r["cost"]) for r in _read_csv(_f(res / "reference_pyvrp_60s.csv"))}


def load_bks(res: Path) -> dict:
    """X name -> BKS cost (from the .sol `Cost` lines)."""
    return {r["name"]: float(r["bks"]) for r in _read_csv(_f(res / "x_records.csv"))}


def load_pomo(res: Path, csv_name: str, scale: float) -> dict:
    """(file-key, mode) -> {pass -> cost}, cost = -reward * scale.

    One POMO column = one mode; its cell pools that mode's tour over the 3
    passes (greedy/dihedral passes 2-3 are byte-identical repeats). The POMO
    ``id`` is the instance number as a string for the CVRP-50 set and the full
    instance name for the X set; numeric ids are re-keyed to the ``cvrp50_%04d``
    file stem (the datagen naming) so POMO joins the reference/solver keys."""
    out: dict = {}
    for r in _read_csv(_f(res / csv_name)):
        key = (f"cvrp50_{int(r['id']):04d}" if r["id"].isdigit() else r["id"], r["mode"])
        out.setdefault(key, {})[int(r["pass"])] = -float(r["reward"]) * scale
    return out


def load_solver_costs(res: Path, method: str) -> dict:
    """(name, budget-column-seconds) -> [cost, ...] (one per committed row).

    PyVRP: one row per (name, budget, seed). OR-Tools: one row per (name, budget)
    (no seed; its repeated runs collapse into the keyed row, so a pooled cell may
    hold < 3 when a sub-second run crashed)."""
    out: dict = {}
    for r in _read_csv(_f(res / SOLVER_CSV[method])):
        out.setdefault((r["file"], float(r["budget"])), []).append(float(r["cost"]))
    return out


# ---------------------------------------------------------------------------
# gap pools (instances x passes)

def _pool_vals(names: list[str], ref: dict, pool: dict, method: str, col: float,
               pomo: dict | None = None, times: dict | None = None) -> list:
    vals = []
    if method == "POMO":
        if times is None:
            return []
        mode = next((m for m in times if abs(times[m] - col) < 1e-9), None)
        for name in names:
            by_id = pomo.get((name, mode), {})
            r0 = ref[name]
            for p in range(3):
                cost = by_id.get(p)
                if cost is not None:
                    vals.append(100.0 * (cost - r0) / r0)
    else:
        for name in names:
            r0 = ref[name]
            row = pool.get((name, col)) or []
            for p in range(3):
                cost = row[p] if p < len(row) else None
                if cost is None:
                    continue
                vals.append(100.0 * (cost - r0) / r0)
    return vals


def pools_for(names, ref, pomo, sol_py, sol_ort, cols, times) -> dict:
    """(method, column-seconds) -> pooled per-instance gap list. POMO's 1 s/10 s
    columns are empty (no column there); its measured columns use mode p's tour
    over the 3 passes. A failing instance/pass contributes nothing."""
    pool = {}
    for method in METHODS:
        for c in cols:
            if method == "POMO":
                pool[(method, c)] = _pool_vals(names, ref, None, method, c, pomo, times)
            elif method == "PyVRP":
                pool[(method, c)] = _pool_vals(names, ref, sol_py, method, c)
            else:
                pool[(method, c)] = _pool_vals(names, ref, sol_ort, method, c)
    return pool


def cell_stats(vals: list):
    """(mean, population-std ddof=0) or None when empty."""
    return None if not vals else (float(np.mean(vals)), float(np.std(vals)))


def _method_rows(pool, cols) -> list[str]:
    rows = []
    for method in METHODS:
        cells = []
        for c in cols:
            st = cell_stats(pool.get((method, c), []))
            cells.append(f"{st[0]:+.2f} ± {st[1]:.2f}" if st else "—")
        rows.append("| " + method + " | " + " | ".join(cells) + " |")
    return rows


# ---------------------------------------------------------------------------
# Environment / Hardware (house format, see results/rl4co_intro/results.md)

def env_hardware(res: Path) -> str:
    def ver(name: str) -> str:
        try:
            return pkg_version(name)
        except Exception:
            return "?"

    smi_line = ""
    p = res / "gpu_sm_query.txt"
    if p.exists():
        smi_line = p.read_text().strip()
    try:
        import torch
        cuda = "True" if torch.cuda.is_available() else "False"
    except Exception:
        cuda = "unavailable"
    smi = (f"- `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`: "
           f"{smi_line}") if smi_line else "- nvidia-smi query line not recorded"
    return "\n".join([
        "## Environment", "",
        f"- Python {sys.version.split()[0]}",
        f"- torch {ver('torch')} (CUDA available: {cuda})",
        f"- rl4co {ver('rl4co')}",
        f"- lightning {ver('lightning')}",
        f"- tensordict {ver('tensordict')}",
        f"- pyvrp {ver('pyvrp')}",
        f"- ortools {ver('ortools')}",
        f"- vrplib {ver('vrplib')}",
        f"- matplotlib {ver('matplotlib')}",
        f"- numpy {ver('numpy')}",
        "",
        "## Hardware", "",
        f"- platform {platform.platform()}",
        f"- nproc {os.cpu_count()}",
        smi,
        "- device used: **cuda**",
    ])


# ---------------------------------------------------------------------------
# section builders

SEED_TABLE = """| Where | Seed(s) | Feeds |
|---|---|---|
| CVRP-50 instance #i | `torch.manual_seed(1234 + i)` | `CVRPGenerator(num_loc=50)([1])` per instance (global torch RNG) |
| POMO sampling budget, pass s | `torch.manual_seed(1000 + s)`, s ∈ {0,1,2} | the sampling start-action RNG |
| PyVRP budget runs | int `seed` arg, 3 seeds (1, 2, 3) | `run_pyvrp.py` CLI → `pyvrp.solve(seed=...)` |
| OR-Tools budget runs | none (CLI has no seed arg) | 3 identical GLS re-runs; std ≈ 0 (documented, not a bug) |
| CVRP-50 reference (PyVRP 60 s) | single fixed seed = 1 | per-instance reference target, not the randomized object |"""


def configuration_md(n50: int, nx: int, times: dict, modes: list) -> str:
    colstr = ", ".join(f"{MODE_LABELS[m]} = {times[m]:g} s" for m in modes)
    return "\n".join([
        "## Configuration", "",
        f"- Columns: the 3 measured POMO-mode times ({colstr}), then the fixed "
        f"`1 s` and `10 s`. POMO fills only its measured columns (its budget is "
        f"the decode time, not wall-clock; its 1 s / 10 s cells are blank and "
        f"drawn as a dotted horizontal extension in the plots).",
        f"- References: CVRP-50 → per-instance PyVRP 60 s (seed = 1); X set → "
        f"BKS from the `.sol` `Cost` lines.",
        f"- Cell = mean ± population std (ddof=0) of the pooled per-instance gap "
        f"values ({n50} × 3 passes / {nx} × 3 passes).",
        "- **±1 note** (once): POMO tours are evaluated in [0,1] float and scaled "
        "×1e6 / ×1000 to the coordinate scale; the solver costs use the nint "
        "integer convention. A ±1 unit offset between the two is expected, not "
        "a bug.",
        "- **OR-Tools std ≈ 0**: the CLI has no seed argument, so its re-runs are "
        "identical deterministic solves; the pooled std is ~0 by construction.",
        "- **POMO-on-X off-distribution caveat**: POMO was trained on [0,1]² "
        "uniform demand-1..10 Q=40; the X instances are far off-distribution, so "
        "its large X gaps (e.g. +260 % greedy on n502) report genuine "
        "distribution shift, not a bug — the PyVRP ≈ BKS cross-check is the "
        "X table's validity anchor.",
        "- **POMO greedy / dihedral passes 2-3 are byte-identical repeats**: "
        "those modes are deterministic per instance, so their 3 passes pool as "
        "repeats (only the sampling pass varies, seeded 1000/1001/1002).",
        "",
        "### Seed summary", "",
        SEED_TABLE,
    ])


# ---------------------------------------------------------------------------
# plots

def _col_label(c: float, times: dict, modes: list) -> str:
    for m in modes:
        if abs(c - times[m]) < 1e-9:
            return f"{MODE_LABELS[m]} ({times[m]:g} s)"
    return f"{int(c)} s" if c == float(int(c)) else f"{c:g} s"


def render_plot(res: Path, pool: dict, times: dict, modes: list, fname: str,
                ref_label: str, title: str) -> Path:
    cols = sorted({times[m] for m in modes} | {1.0, 10.0})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")
    for method in METHODS:
        keys = [times[m] for m in modes] if method == "POMO" else cols
        xs, ms, ss = [], [], []
        for c in keys:
            st = cell_stats(pool.get((method, c), []))
            if st is None:
                continue
            xs.append(c); ms.append(st[0]); ss.append(st[1])
        if not xs:
            continue
        col = METHOD_COLOR[method]
        ax.plot(xs, ms, marker="o", color=col, label=method)
        ax.fill_between(xs, [a - b for a, b in zip(ms, ss)],
                        [a + b for a, b in zip(ms, ss)],
                        color=col, alpha=0.15, lw=0)
        # POMO: extend from its last point to the right edge (budget is decode-time)
        if method == "POMO" and xs and max(times.values()) < 10:
            ax.plot([xs[-1], cols[-1]], [ms[-1], ms[-1]], color=col,
                    linestyle="dotted", lw=1.5, alpha=0.8)
    ax.set_xticks(cols, labels=[_col_label(c, times, modes) for c in cols],
                  rotation=30)
    ax.set_xlabel("time budget (s, log scale)")
    ax.set_ylabel(f"gap vs {ref_label} (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    cap = ("Mean gap ± 1 population std of the pooled per-instance gaps; POMO's "
           "budget is its own decode time (not wall-clock), hence the dotted "
           "horizontal extension.")
    ax.set_title(title)
    fig.text(0.5, 0.01, cap, ha="center", fontsize=8)
    fig.tight_layout()
    out = res / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# main

def main(res_dir: Path) -> Path:
    res = Path(res_dir)
    times, modes = load_budgets(res)
    ref50 = load_reference(res)
    bks = load_bks(res)
    pomo50 = load_pomo(res, "pomo_cvrp50.csv", 1e6)
    pomo_x = load_pomo(res, "pomo_x.csv", 1e3)
    sol_py = load_solver_costs(res, "PyVRP")
    sol_ort = load_solver_costs(res, "OR-Tools")

    cols = sorted({times[m] for m in modes} | {1.0, 10.0})
    pool50 = pools_for(sorted(ref50), ref50, pomo50, sol_py, sol_ort, cols, times)
    poolx = pools_for(sorted(bks), bks, pomo_x, sol_py, sol_ort, cols, times)

    header = "| method | " + " | ".join(_col_label(c, times, modes) for c in cols) + " |"
    sep = "|---|" * (len(cols) + 1)
    table50 = [header, sep] + _method_rows(pool50, cols)
    tablex = [header, sep] + _method_rows(poolx, cols)

    render_plot(res, pool50, times, modes, "gap_time_cvrp50.png",
                "PyVRP 60 s reference", "Gap vs PyVRP 60 s reference — CVRP-50 set")
    render_plot(res, poolx, times, modes, "gap_time_x.png",
                "best-known solution", "Gap vs BKS — X set")

    empty50 = [f"{m} @ {c:g} s" for m in METHODS for c in cols if not pool50[(m, c)]]
    emptyx = [f"{m} @ {c:g} s" for m in METHODS for c in cols if not poolx[(m, c)]]
    if empty50 or emptyx:
        print(f"aggregate: empty cells  cvrp50={empty50}  x={emptyx}")
    empty_note = ""
    if t_empty := (empty50 or emptyx):
        empty_note = "\n" + "- Empty cells (no committed row; logged): " + \
            f"CVRP-50 {empty50 or '∅'}; X set {emptyx or '∅'}. (OR-Tools sub-second " \
            "cells can be empty on rc=1 `AttributeError`.)"

    md = "\n".join([
        "# Equal-budget evaluation — POMO vs PyVRP vs OR-Tools (CVRP-50 + X set)",
        "",
        env_hardware(res),
        "",
        "## Tables (gap vs reference, %)",
        "",
        f"### CVRP-50 set ({len(ref50)} instances × 3 passes)",
        "",
        *table50,
        empty_note,
        "",
        f"### X set ({len(bks)} instances × 3 passes)",
        "",
        *tablex,
        "",
        "*Plots: `gap_time_cvrp50.png`, `gap_time_x.png`.*",
        "",
        configuration_md(len(ref50), len(bks), times, modes),
    ])
    out = res / "results.md"
    out.write_text(md)
    print(f"aggregate: wrote results.md (cols={cols}) and 2 PNGs "
          f"(empty50={empty50}, emptyx={emptyx})")
    return out


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2] / "results" / "equal_evaluation")
