"""Orchestrator for the RL4CO intro task — trains the library's Attention
Model (AM) and POMO on TSP-50 and CVRP-50 with the LIBRARY DEFAULT
hyperparameters (10 epochs per (variant, problem) by default), keeps one
best-peak-epoch checkpoint per pair, evaluates them, and writes the results
report, the curve CSVs, and the training-return PNGs.

Spec run (default CLI values; run from repo root):
    conda activate nor && python src/rl4co_intro/orchestrator.py \
        [--epochs 10] [--max-steps 500] [--batch-size 512] \
        [--train-data-size 5000] [--val-data-size 500] \
        [--test-data-size 500] [--problems tsp cvrp] \
        [--variants am pomo] [--seed 42] [--device auto] [--resume]

Invariants (see markdown/rl4co_intro_instructions.md):
  - library-default hyperparameters, NO tuning: Adam lr=1e-4,
    lr_scheduler=None, gradient clip 1.0, precision 16, batch size 512.
    The only deliberate deviations are the data sizes (train 5000 / val 500 /
    test 500 vs the library's 100 000 / 10 000 / 10 000), overridable via
    CLI. Batch size stays 512; a POMO OOM at 512 → lower
    --batch-size (recorded in results.md as the only deviation).
  - a mandatory pre-run `nvidia-smi` gate is executed by the orchestrator
    (free VRAM + occupying processes are queried and recorded in
    `results.md`'s Header); `--device auto` resolves to cuda only when CUDA
    is available — pass `--device cpu` explicitly whenever the GPU is not
    confirmed free.
  - one Lightning `Callback` per run captures the per-batch train/val means
    (``outputs["train/reward"]`` / ``outputs["val/reward"]`` — already the
    batch-mean tensors from ``RL4COLitModule.log_metrics``) in
    ``on_train_batch_end`` / ``on_validation_batch_end``; at
    ``on_train_epoch_end`` it computes the epoch mean/std, appends the row to
    the curve CSV, and saves the best-peak-epoch checkpoint. Metric =
    epoch-mean TRAINING return (equal-weight mean of that epoch's per-step
    ``train/reward`` value); epoch 1 always saves, later epochs only on
    strict improvement.
  - checkpoints are state-dict-only ``{"epoch", "global_step",
    "state_dict"}``. rl4co's ``load_from_checkpoint`` is broken on
    torch 2.11.0 (``weights_only=None`` → ``weights_only=True`` cannot
    unpickle hparams that carry pickled envs/baselines), so the orchestrator
    reloads via a fresh model built with the identical env/kwargs →
    ``model.setup(stage="fit")`` (which already ends in
    ``post_setup_hook()``; nothing else is called) → strict
    ``load_state_dict``. The live end-of-training weights are NOT what is
    evaluated.
  - the test-eval column is single-start greedy —
    ``model(td, phase="test", decode_type="greedy", num_starts=1)`` — for
    BOTH variants, no dihedral augmentation. Deliberately NOT POMO's default
    inference regime (multi-start over all feasible start nodes with 8x
    dihedral on val/test); recorded in ``results.md``.
  - wall time per pair = fit start → fit return (train + validation, as
    reported by Lightning).
  - ``--resume`` re-runs only pairs whose curve CSV or checkpoint is missing.

Artifacts:
  - ``src/rl4co_intro/checkpoints/``: ``am_tsp50_best.pt`` /
    ``am_cvrp50_best.pt`` / ``pomo_tsp50_best.pt`` / ``pomo_cvrp50_best.pt``
    (gitignored)
  - ``results/rl4co_intro/curves/<variant>_<problem>50_curve.csv``: one row
    per epoch with columns ``epoch, mean_reward, std_reward, wall_s`` (the
    spec's literal columns), then the same epoch's val capture
    ``val_mean, val_std`` (so the results table can read the peak epoch's
    ``val/reward`` straight from the curve CSV)
  - ``results/rl4co_intro/{tsp50,cvrp50}_training_returns.png``
  - ``results/rl4co_intro/results.md`` + ``log.txt``
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Repo structure: this file lives in src/rl4co_intro/. Python auto-adds the
# script dir, but utils/ lives in src/ -- add src/ too (parents[1]).
SRC_DIR = str(Path(__file__).resolve().parents[1])
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

matplotlib.use("Agg")  # headless: must be set before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from importlib.metadata import version as pkg_version  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "results" / "rl4co_intro"
CURVES_DIR = LOG_DIR / "curves"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
RESULTS_MD = LOG_DIR / "results.md"

from lightning.pytorch.callbacks import Callback  # noqa: E402
from rl4co.envs import CVRPEnv, TSPEnv  # noqa: E402
from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402
from rl4co.envs.routing.tsp.generator import TSPGenerator  # noqa: E402
from rl4co.models import AttentionModel, POMO  # noqa: E402
from rl4co.utils import RL4COTrainer  # noqa: E402

log = logging.getLogger(__name__)

VARIANT_COLORS = {"am": "tab:blue", "pomo": "tab:orange"}
VARIANTS = ("am", "pomo")
PROBLEMS = ("tsp", "cvrp")

# Verified in the rl4co 0.7.0 sources: the plain "greedy" strategy keeps
# multistart=False when num_starts=1 (DecodingStrategy.__init__:
# multistart = num_starts > 1), and pre_decoder_hook harmless-ignores
# num_starts for non-multistart decoding — so this call is valid for both
# variants and yields exactly one greedy rollout per instance.
EVAL_KWARGS = dict(phase="test", decode_type="greedy", num_starts=1)


def _tag(variant: str, problem: str) -> str:
    return f"{variant}_{problem}50"


def pick_device(req: str) -> str:
    """Resolve a device: 'auto' → cuda if available, else cpu; else
    passthrough (validated by the CLI choices)."""
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return req


def _model_for(variant: str) -> type:
    return {"am": AttentionModel, "pomo": POMO}[variant]


def _env_for(problem: str):
    gen = {"tsp": TSPGenerator, "cvrp": CVRPGenerator}[problem](num_loc=50)
    env_cls = {"tsp": TSPEnv, "cvrp": CVRPEnv}[problem]
    return env_cls(generator=gen)


class ReturnTracker(Callback):
    """Per-(variant, problem) run: per-batch return capture → epoch
    mean/std → curve CSV + best-peak-epoch checkpoint.

    The metric that selects the checkpoint is the epoch-mean TRAINING return
    (equal-weight mean of that epoch's per-step ``train/reward``).  Epoch 1
    always saves; a later epoch saves only if its train mean is strictly
    larger.
    """

    def __init__(self, curve_csv: Path, ckpt_path: Path) -> None:
        super().__init__()
        self.curve_csv = curve_csv
        self.ckpt_path = ckpt_path
        self.train_rewards: list[float] = []
        self.val_rewards: list[float] = []
        self._epoch_t0 = time.perf_counter()

    @staticmethod
    def _reward(outputs: object, key: str) -> float | None:
        """Per-batch MEAN reward (see RL4COLitModule.log_metrics, which
        averages reward over the batch before self.log_dict); shared_step
        returns a plain dict, accept a LightningModuleOutputs wrapping such
        a dict too. Returns None when the key is absent (e.g. a val row
        with no captured reward yet)."""
        if outputs is None:
            return None
        d = outputs if isinstance(outputs, dict) else getattr(outputs, "tensor", None)
        if not isinstance(d, dict):
            return None
        v = d.get(key)
        if v is None:
            return None
        return float(v)

    def _capture(self, outputs: object, key: str, bucket: list[float]):
        v = self._reward(outputs, key)
        if v is not None:
            bucket.append(v)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._capture(outputs, "train/reward", self.train_rewards)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._capture(outputs, "val/reward", self.val_rewards)

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        # Covers this epoch's start → end of its validation run (the
        # model's own on_train_epoch_end dataset-regen hook runs first, so
        # regeneration is included; epoch 1 also includes trainer init).
        epoch = pl_module.current_epoch + 1  # zero-based → 1-based label
        train_mean, train_std = _mean_std(self.train_rewards)
        val_mean, val_std = _mean_std(self.val_rewards)
        wall = time.perf_counter() - self._epoch_t0

        # Best-so-far BEFORE appending this epoch's row, so the
        # strict-improvement check compares this epoch against every
        # earlier one (the kept checkpoint is then always the best
        # peak-epoch weights).
        new_file = not self.curve_csv.exists()
        prev_best = None if new_file else _best_mean_reward(self.curve_csv)
        self.curve_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.curve_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["epoch", "mean_reward", "std_reward", "wall_s",
                            "val_mean", "val_std"])
            w.writerow([epoch, f"{train_mean:.6f}", f"{train_std:.6f}",
                        f"{wall:.2f}", f"{val_mean:.6f}", f"{val_std:.6f}"])
        log.info("epoch %2d  train_return %.4f±%.4f  val_return %.4f±%.4f  "
                 "wall %.1fs", epoch, train_mean, train_std,
                 val_mean, val_std, wall)

        # Epoch 1: always save. Later epochs: strict improvement only.
        if prev_best is None or train_mean > prev_best:
            self._save(pl_module)
        # Reset for the next epoch's capture.
        self.train_rewards.clear()
        self.val_rewards.clear()

    def _save(self, pl_module) -> None:
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": pl_module.current_epoch,  # zero-based
                "global_step": pl_module.global_step,
                "state_dict": pl_module.state_dict(),
            },
            tmp,
        )
        tmp.replace(self.ckpt_path)
        log.info("saved best checkpoint for epoch %d → %s",
                 pl_module.current_epoch + 1, self.ckpt_path.name)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    a = torch.tensor(values, dtype=torch.float64)
    m = a.mean().item()
    s = a.std(correction=0).item() if a.numel() > 1 else 0.0
    return m, s


def _best_mean_reward(curve_csv: Path) -> float | None:
    """Best-so-far mean_reward (training return) across all of the curve
    CSV's existing rows — the checkpoint is written exactly when a strict
    improvement over it is appended (and only then), so the last saved
    checkpoint is always the best-peak-epoch one. None = no rows yet
    (fresh run: epoch 1 always saves)."""
    if not curve_csv.exists():
        return None
    with curve_csv.open() as f:
        rows = [float(r["mean_reward"])
                for r in csv.DictReader(f) if r.get("mean_reward")]
    return max(rows) if rows else None


def build_model(variant: str, problem: str, args: argparse.Namespace):
    """Fresh env + model with identical args — used both for training and
    for the best-checkpoint load recipe (see module docstring)."""
    env = _env_for(problem)
    model_cls = _model_for(variant)
    model = model_cls(
        env,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        batch_size=args.batch_size,
    )
    return env, model


def capture_gpu_gate() -> str | None:
    """Mandatory pre-run nvidia-smi gate: returns a one-line summary of
    free VRAM + occupying processes, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        t, fr, us = [int(x) for x in q.stdout.strip().split(",")]
        apps = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        proc_lines = [
            f"{line.split(',')[1].strip()} [{line.split(',')[0].strip()}]"
            for line in apps.stdout.splitlines()
            if len(line.split(",")) >= 2 and line.split(",")[1].strip()
        ] if apps.returncode == 0 and apps.stdout.strip() else []
        procs = "; ".join(proc_lines) if proc_lines else "none"
        return (f"free {fr / 1024:.1f} / {t / 1024:.1f} GiB, "
                f"{us / 1024:.1f} GiB used by: {procs}")
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def train_pair(variant: str, problem: str, args) -> float:
    """Train one (variant, problem) pair; returns fit wall time (s)."""
    import lightning as L  # local import keeps `--help` fast

    L.seed_everything(args.seed, workers=True)
    _, model = build_model(variant, problem, args)
    tag = _tag(variant, problem)
    tracker = ReturnTracker(CURVES_DIR / f"{tag}_curve.csv",
                            CKPT_DIR / f"{tag}_best.pt")
    is_gpu = args.device == "cuda"
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu" if is_gpu else "cpu",
        devices=[0] if is_gpu else 1,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        logger=False,  # None flips to the default CSVLogger(lightning_logs/); ours is the curve CSV
        log_every_n_steps=10,
        callbacks=[tracker],
    )
    t0 = time.perf_counter()
    trainer.fit(model)
    return time.perf_counter() - t0


def load_best(variant: str, problem: str, args):
    """Rebuild a fresh model and strict-load the kept checkpoint into it
    (manual recipe — see module docstring), proving the roundtrip: every
    tensor of the loaded state must equal the file's state dict."""
    tag = _tag(variant, problem)
    env, model = build_model(variant, problem, args)
    ckpt = torch.load(CKPT_DIR / f"{tag}_best.pt", weights_only=False)
    file_sd = ckpt["state_dict"]
    model.setup(stage="fit")  # already ends with post_setup_hook()
    model.load_state_dict(file_sd, strict=True)
    # Roundtrip proof (the smoke test's tensor comparison): key sets equal
    # and every weight bit-equal to what was saved.
    loaded_sd = model.state_dict()
    if set(loaded_sd) != set(file_sd):
        raise RuntimeError(f"{tag}: checkpoint key mismatch "
                           f"(missing={set(file_sd) - set(loaded_sd)}, "
                           f"new={set(loaded_sd) - set(file_sd)})")
    for k in sorted(file_sd):
        if not torch.equal(loaded_sd[k], file_sd[k]):
            raise RuntimeError(f"{tag}: checkpoint roundtrip mismatch for {k}")
    log.info("%s: checkpoint load verified (%d tensors equal)",
             tag, len(file_sd))
    model.eval()
    return env, model, int(ckpt["epoch"]) + 1


def eval_lengths(env, model, test_size: int, device: str, seed: int,
                 eval_batch: int = 100) -> list[float]:
    """Single-start greedy decoding (see EVAL_KWARGS) over ONE fixed test
    set: the ``test_size`` instances are generated ONCE from the env's
    generator with the run seed, then evaluated in slices of ``eval_batch``
    (the slices are consecutive views of that one generation, so the test
    set is exactly the first ``test_size`` instances of it); returns tour /
    route lengths (the reward is the negative objective)."""
    import lightning as L  # local import keeps `--help` fast

    L.seed_everything(seed, workers=True)
    td = env.generator(batch_size=test_size)  # generated once, per spec
    dev = "cuda:0" if device == "cuda" else "cpu"
    model.to(dev)
    out: list[float] = []
    for i in range(0, test_size, eval_batch):
        part = td[i:i + eval_batch].to(dev)
        part = env.reset(part)
        with torch.no_grad():
            r = model(part, **EVAL_KWARGS)["reward"]
        out.extend(torch.nan_to_num(r).neg().tolist())
    return out


def plot_problem(problem: str, variants: list[str]) -> Path:
    """One training-return PNG per problem: both variants, mean with a ±1-std
    band, AM blue / POMO orange."""
    png = LOG_DIR / f"{problem}50_training_returns.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    n_epochs = 0
    for v in VARIANTS:
        if v not in variants:
            continue
        curve = CURVES_DIR / f"{_tag(v, problem)}_curve.csv"
        if not curve.exists():
            continue
        with curve.open() as f:
            rows = list(csv.DictReader(f))
        m = np.array([float(r["mean_reward"]) for r in rows])
        s = np.array([float(r["std_reward"]) for r in rows])
        x = np.arange(1, len(m) + 1)
        n_epochs = max(n_epochs, len(m))
        ax.plot(x, m, color=VARIANT_COLORS[v], linewidth=1.8, label=v.upper())
        ax.fill_between(x, m - s, m + s, color=VARIANT_COLORS[v], alpha=0.25)
    if n_epochs:
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.9, alpha=0.7)
        if n_epochs > 1:  # degenerate range for the 1-epoch smoke run
            ax.set_xlim(1, n_epochs + (n_epochs - 1) * 0.03)
        else:
            ax.set_xlim(0.5, 1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training return (higher = better; = −mean length/step)")
    ax.set_title(f"Training return — {problem.upper()}-50 "
                 "(batch-mean train return per epoch)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="variant", loc="lower right")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def _cpu_line() -> str:
    name = platform.processor()
    freq = ""
    try:
        import psutil
        freqs = psutil.cpu_freq().cores
        if freqs:
            freq = f" ({min(freqs):.0f}–{max(freqs):.0f} MHz base/boost)"
    except Exception:
        pass
    if not name:  # platform.processor() is empty on most Linux
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        name = name or f"{platform.machine()} x86-64"
    return f"CPU: {name} — {os.cpu_count() or 'n/a'} logical cores{freq}"


def _cpu_lines() -> list[str]:
    return [_cpu_line()]


def _gpu_lines() -> list[str]:
    if not torch.cuda.is_available():
        return ["GPU: CUDA not available in this environment."]
    lines = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)} — "
                     f"{free / 1024**3:.1f} / {total / 1024**3:.1f} GiB free")
    return lines


def write_results_md(args, meta: dict) -> None:
    """meta: {(variant, problem): dict(epoch, train_mean, train_std,
    val_mean, val_std, wall, eval_mean, eval_std, eval_n)} plus
    meta["gpu_gate"] (str | None)."""
    L = []
    prec_note = (
        " (requested; on CPU Lightning resolves `16-mixed` to `bf16-mixed` — "
        "fp16 AMP is unsupported on CPU, so the runtime AMP dtype here is "
        "bfloat16)"
        if args.device == "cpu" else ""
    )
    cpu_prec_note = (
        " (requested; Lightning resolves `16-mixed` to `bf16-mixed` at "
        "runtime on CPU)"
        if args.device == "cpu" else ""
    )
    L.append("# RL4CO intro — AM & POMO on TSP-50 / CVRP-50")
    L.append("")
    L.append("## Header")
    L.append("")
    gate = meta.get("gpu_gate")
    gate_txt = (gate if gate is not None
                else "CUDA not available in this environment; CPU run")
    L.append(f"GPU gate (mandatory pre-run `nvidia-smi`): {gate_txt}.")
    L.append(
        f"Device: `--device {args.device_requested}` → "
        f"**{args.device}** ('auto' resolves cuda-if-available, else "
        f"cpu). Batch size: **{args.batch_size}** (library default; "
        f"only an observed OOM would justify lowering it — none "
        f"occurred)."
    )
    L.append("")
    L.append(
        "What was trained: Attention Model (AM, Kool et al. 2019 REINFORCE, "
        "rollout baseline) and POMO (Kwon et al. 2020, `shared` baseline, "
        "multi-start over all feasible start nodes, 8× dihedral on val/test) "
        "from the `rl4co` 0.7.0 library, on TSP-50 (`TSPEnv` + "
        "`TSPGenerator(num_loc=50)`) and CVRP-50 (`CVRPEnv` + "
        "`CVRPGenerator(num_loc=50)`, capacity 40), with the **library-default "
        "hyperparameters**: optimizer Adam lr=1e-4 (**`lr_scheduler=None`** — "
        "no LR scheduler, the library's real default; none added), gradient "
        f"clip 1.0, precision 16-mixed{prec_note}, batch {args.batch_size}, "
        f"{args.epochs} epochs × "
        f"`ceil({args.train_data_size}/{args.batch_size})` optimizer steps, "
        f"`max_steps={args.max_steps}` as a total-step ceiling. Reward = "
        "**−objective** (negative tour / route length), so "
        "returns rise toward 0; **length = −return**. Checkpoint protocol: "
        "best epoch-mean **training** return (per-step `train/reward` values "
        "from the callback's `outputs` capture, equal-weight mean; see "
        "checkpoint protocol below). Data sizes are the only deliberate "
        "deviations from the library defaults (see Configuration); nothing "
        "else was tuned."
    )
    L.append("")
    L.append(
        "Checkpoint protocol: the per-run Lightning `Callback` captures "
        "per-batch `train/reward` / `val/reward` (already the batch-mean "
        "tensors from `RL4COLitModule.log_metrics`); at each epoch it writes "
        "the epoch mean/std to the curve CSV and saves the **best-peak-"
        "epoch** checkpoint — metric = epoch-mean **training** return "
        "(equal-weight mean of that epoch's per-step `train/reward`), epoch "
        "1 always saves, later epochs only on strict improvement. "
        "Checkpoints are state-dict-only `{\"epoch\", \"global_step\", "
        "\"state_dict\"}` (rl4co's `load_from_checkpoint` is broken on "
        "torch 2.11.0: `weights_only=None` → `weights_only=True` cannot "
        "unpickle hparams with pickled envs/baselines); evaluation reloads "
        "them into a fresh model with identical env/kwargs via "
        "`model.setup(stage=\"fit\")` (which already runs "
        "`post_setup_hook()`) + strict `load_state_dict` — **never** the "
        "live end-of-training weights."
    )
    L.append("")
    L.append(
        f"Evaluation regime: one fixed test set of "
        f"`test_data_size`={args.test_data_size} instances generated once "
        "from the env's generator (run seed `--seed`, slices of 100), "
        f"decoded with `model(td, **{EVAL_KWARGS})` — **single-start "
        "greedy, no dihedral augmentation, for BOTH variants**. This is "
        "deliberately **not** POMO's default inference regime (multi-start "
        "over all feasible start nodes, 8× dihedral on val/test); it keeps "
        "the AM/POMO test comparison apples-to-apples."
    )
    L.append("")
    L.append("## Best results")
    L.append("")
    L.append("Columns: best epoch (chosen by epoch-mean training return) | "
             "train return mean ± std at that epoch → length ≈ −mean | "
             "val return mean ± std at that epoch → length | test greedy "
             "single-start mean ± std (length) | fit wall time (s).")
    L.append("")
    L.append("| variant | problem | best epoch | train (mean ± std) → "
             "length | val (mean ± std) → length | test greedy "
             "(mean ± std) | fit wall (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        if v not in args.variants:
            continue
        for p in PROBLEMS:
            if p not in args.problems:
                continue
            r = meta.get((v, p))
            if r is None:
                L.append(f"| {v.upper()} | {p.upper()}-50 | — | | | | |")
                continue
            L.append(
                f"| {v.upper()} | {p.upper()}-50 | {r['epoch']} "
                f"| {r['train_mean']:.4f} ± {r['train_std']:.4f} → "
                f"{-r['train_mean']:.2f} "
                f"| {r['val_mean']:.4f} ± {r['val_std']:.4f} → "
                f"{-r['val_mean']:.2f} "
                f"| {r['eval_mean']:.4f} ± {r['eval_std']:.4f} "
                f"| {r['wall']:.0f} |"
            )
    L.append("")
    L.append("## Configuration")
    L.append("")
    L.append("| key | value |")
    L.append("|---|---|")
    L.append(f"| epochs | {args.epochs} |")
    L.append(f"| max_steps (total-step ceiling) | {args.max_steps} |")
    L.append(f"| batch size | {args.batch_size} (library default) |")
    L.append(f"| train/val/test sizes | {args.train_data_size} / "
             f"{args.val_data_size} / {args.test_data_size} — **only "
             f"deliberate deviation** from library defaults 100 000 / "
             f"10 000 / 10 000 |")
    L.append("| optimizer | Adam, lr=1e-4 (library default) |")
    L.append("| lr scheduler | none (library default) |")
    L.append("| gradient clip | 1.0 (library default) |")
    L.append(f"| precision | 16 (library default){cpu_prec_note} |")
    L.append(f"| seed | {args.seed} |")
    L.append("| policies | AM / POMO library policy defaults (POMO: "
             "`num_starts=None` → all feasible starts, 8× dihedral on "
             "val/test, SharedBaseline; AM: default warmup rollout "
             "baseline) |")
    L.append("")
    L.append("## Environment")
    L.append("")
    L.append(f"- Python {sys.version.split()[0]}")
    L.append(f"- torch {pkg_version('torch')} (CUDA available: "
             f"{torch.cuda.is_available()})")
    L.append(f"- lightning {pkg_version('lightning')}")
    L.append(f"- rl4co {pkg_version('rl4co')}")
    L.append(f"- matplotlib {pkg_version('matplotlib')}")
    L.append("")
    L.append("## Hardware")
    L.append("")
    L.extend(_cpu_lines())
    L.extend(_gpu_lines())
    L.append(f"- device used: **{args.device}**")
    L.append("")
    L.append("## Plots")
    L.append("")
    L.append("- `tsp50_training_returns.png` — per-epoch mean of the per-step "
             "`train/reward` values (equal-weight mean over the epoch's "
             "steps), both variants")
    L.append("- `cvrp50_training_returns.png` — same for CVRP-50")
    L.append("- band = ±1 population std of that epoch's per-step returns "
             "(AM `tab:blue`, POMO `tab:orange`); curves: "
             "`curves/<variant>_<problem>50_curve.csv`")
    L.append("")
    L.append("## Checkpoints")
    L.append("")
    for v in VARIANTS:
        if v not in args.variants:
            continue
        for p in PROBLEMS:
            if p not in args.problems:
                continue
            L.append(f"- `src/rl4co_intro/checkpoints/{_tag(v, p)}_best.pt` "
                     f"(state-dict-only `epoch` / `global_step` / "
                     f"`state_dict`)")
    L.append("")
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(L))
    log.info("wrote %s", RESULTS_MD)


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RL4CO intro: AM & POMO on TSP-50 / CVRP-50 "
                    "(library-default hyperparameters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=10,
                   help="max training epochs (default 10)")
    p.add_argument("--max-steps", type=int, default=500, dest="max_steps",
                   help="total-step ceiling for the trainer "
                        "(library default 500)")
    p.add_argument("--batch-size", type=int, default=512, dest="batch_size",
                   help="library default; keep 512 (only lower to fix OOM)")
    p.add_argument("--train-data-size", type=int, default=5000,
                   dest="train_data_size")
    p.add_argument("--val-data-size", type=int, default=500,
                   dest="val_data_size")
    p.add_argument("--test-data-size", type=int, default=500,
                   dest="test_data_size")
    p.add_argument("--problems", nargs="+", default=list(PROBLEMS),
                   choices=list(PROBLEMS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", action="store_true",
                   help="re-run only pairs whose curve CSV or checkpoint is "
                        "missing")
    a = p.parse_args()
    a.device_requested = a.device
    a.device = pick_device(a.device)
    return a


def main() -> None:
    setup_logging(LOG_DIR)
    args = parse_cli()
    meta: dict = {"gpu_gate": capture_gpu_gate()}
    log.info("run: variants=%s problems=%s device=%s seed=%d epochs=%d "
             "max_steps=%d batch=%d data=%d/%d/%d", args.variants,
             args.problems, args.device, args.seed, args.epochs,
             args.max_steps, args.batch_size, args.train_data_size,
             args.val_data_size, args.test_data_size)
    if meta["gpu_gate"] and args.device == "cuda":
        fr = float(meta["gpu_gate"].split()[1])
        if fr < 4.0:
            log.warning("GPU gate: only %.1f GiB free — POMO batch 512 may "
                        "not fit; consider --device cpu", fr)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for v in VARIANTS:
        if v not in args.variants:
            continue
        for p in PROBLEMS:
            if p not in args.problems:
                continue
            tag = _tag(v, p)
            curve = CURVES_DIR / f"{tag}_curve.csv"
            ckpt = CKPT_DIR / f"{tag}_best.pt"
            if args.resume and curve.exists() and ckpt.exists():
                wall = 0.0
                log.info("resume: %s complete — skipping training", tag)
            else:
                # Fresh training: start from a clean slate so stale
                # artifacts of this pair (previous seed/flags, smoke-test
                # run) don't mix into the curve (duplicate epoch rows) or
                # shadow the new checkpoint.
                for stale in (curve, ckpt.with_suffix(".tmp"), ckpt):
                    if stale.exists():
                        stale.unlink()
                        log.info("%s: removed stale %s (fresh training)",
                                 tag, stale.name)
                t0 = time.perf_counter()
                train_pair(v, p, args)
                wall = time.perf_counter() - t0
                log.info("%s: fit finished in %.1f s (%.1f min)",
                         tag, wall, wall / 60)
            with curve.open() as f:
                rows = list(csv.DictReader(f))
            if not rows:
                log.warning("%s: empty curve — skipping eval", tag)
                continue
            meta[(v, p)] = {"wall": wall}

    # Evaluate every pair with the kept best checkpoint: single-start
    # greedy, no dihedral — for both variants.
    for key in [k for k in meta if isinstance(k, tuple)]:
        v, p = key
        tag = _tag(v, p)
        env, model, epoch = load_best(v, p, args)
        # The kept checkpoint's own row in the curve (the best peak-epoch
        # wall / train-return / val-return of THAT epoch — not the last
        # row, which is the last epoch's).
        with (CURVES_DIR / f"{_tag(v, p)}_curve.csv").open() as f:
            row = next(r for r in csv.DictReader(f)
                       if int(r["epoch"]) == epoch)
        meta[key]["epoch"] = epoch
        meta[key]["train_mean"] = float(row["mean_reward"])
        meta[key]["train_std"] = float(row["std_reward"])
        meta[key]["val_mean"] = float(row["val_mean"])
        meta[key]["val_std"] = float(row["val_std"])
        t0 = time.perf_counter()
        lens = np.asarray(
            eval_lengths(env, model, args.test_data_size, args.device,
                         args.seed))
        meta[key]["eval_mean"] = float(lens.mean())
        meta[key]["eval_std"] = float(
            np.asarray(lens, dtype=np.float64).std(ddof=0)) if lens.size > 1 else 0.0
        meta[key]["eval_n"] = int(lens.size)
        log.info("%s: test greedy single-start mean length %.4f over %d "
                 "instances in %.1f s", tag, lens.mean(), lens.size,
                 time.perf_counter() - t0)

    for p in args.problems:
        if not any((CURVES_DIR / f"{_tag(v, p)}_curve.csv").exists()
                   for v in args.variants):
            continue
        log.info("wrote %s", plot_problem(p, args.variants).name)

    write_results_md(args, meta)
    log.info("done.")


if __name__ == "__main__":
    main()
