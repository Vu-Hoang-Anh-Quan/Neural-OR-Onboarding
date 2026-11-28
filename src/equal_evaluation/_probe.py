"""Temporary de-risk probe (deleted after the smoke gate). Exercises, without any model:
  1. CVRPGenerator(50) seed round-trip for instance #0 + TensorDictDataset collate.
  2. Env round-trip on the CVRP-50 td (reset/step semantics, scripted tour, get_reward).
  3. Hand-built X-n502-k39 td + scripted tour + env.get_reward + independent raw recheck.
  4. GreedyEval (the eval-API surface used via evaluate_policy) with a fake policy.
  5. run_pyvrp.py subprocess (2 s, seed 1) on n502 for the BKS cross-check.
Run: /home/Quan/miniconda3/envs/nor/bin/python src/equal_evaluation/_probe.py
"""

import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from cvrp.cvrp_utils import load  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402

from rl4co.data.dataset import TensorDictDataset  # noqa: E402
from rl4co.envs import CVRPEnv  # noqa: E402
from rl4co.envs.routing.cvrp.generator import CVRPGenerator  # noqa: E402
from rl4co.tasks.eval import GreedyEval  # noqa: E402

setup_logging(REPO_ROOT / "results" / "equal_evaluation" / "_probe")

DE = 1e-5
N_X = 501
Q_X = 13
BKS_X = 69226
X_FILE = REPO_ROOT / "data" / "chosen" / "X-n502-k39.vrp"


def scripted_tour(env, td_reset):
    """Feasibility-greedy tour (env masks) through the real env.step. Returns actions [N+K+1]."""
    acts = []
    t = td_reset.clone()
    while not t["done"].all():
        a = t["action_mask"].float().argmax(-1).long()
        t = env.step(t.set("action", torch.full((1,), int(a), dtype=torch.long, device=t.device)))["next"]
        acts.append(int(a))
    return torch.tensor(acts, device=td_reset.device).unsqueeze(0)  # [1, N] (2D, as policy actions)


def raw_tour_len(actions, xy, demand01, q01):
    """Independent (no-tensor) tour length + raw capacity check.

    `actions` is the full policy tour INCLUDING depot returns (0). Routes are
    reconstructed at the actual depot actions, and each route's EUC leg length is
    summed with float coords (no nint). Also returns (L, routes) where routes is a
    list of customer-index lists for the raw capacity check.
    """
    routes, cur = [], []
    for a in actions:
        if a == 0:
            if cur:
                routes.append(cur)
                cur = []
        else:
            cur.append(a)
    if cur:
        routes.append(cur)
    p0 = torch.tensor([p[0] for p in xy])
    p1 = torch.tensor([p[1] for p in xy])
    locs2 = torch.stack([p0, p1], -1)

    def d(i, j):
        return float((locs2[i] - locs2[j]).square().sum().sqrt())

    L = 0.0
    for r in routes:
        seq = [0] + r + [0]
        L += sum(d(seq[k], seq[k + 1]) for k in range(len(seq) - 1))
    return L, routes


# --- 1. CVRP-50 seed round-trip ---------------------------------------------
torch.manual_seed(1234 + 0)
gen = CVRPGenerator(num_loc=50)
td0 = gen([1])
assert tuple(td0.batch_size) == (1,)
assert tuple(td0["locs"].shape) == (1, 50, 2)
assert tuple(td0["depot"].shape) == (1, 2)
assert tuple(td0["demand"].shape) == (1, 50)
assert tuple(td0["capacity"].shape) == (1, 1)
assert torch.isclose(td0["capacity"][0, 0], torch.tensor(40.0)).item()
raw0 = (td0["demand"][0].clamp(0, 1) * 40).round().long()
print("#0: raw demand min/max/sum", int(raw0.min()), int(raw0.max()), int(raw0.sum()),
      "| locs", float(td0["locs"].min()), float(td0["locs"].max()))

ds = TensorDictDataset(td0)
loader = DataLoader(ds, batch_size=1, collate_fn=TensorDictDataset.collate_fn)
batch = next(iter(loader))
assert tuple(batch.batch_size) == (1,) and set(batch.keys()) == {"capacity", "depot", "demand", "locs"}
print("TensorDictDataset collate OK:", sorted(batch.keys()))

# --- 2. env round-trip on CVRP-50 #0 ---------------------------------------
env = CVRPEnv(generator=gen)
tdr0 = env.reset(td0.clone())
assert tuple(tdr0["locs"].shape) == (1, 51, 2)
assert torch.equal(tdr0["locs"][0, 0], td0["depot"][0])
acts0 = scripted_tour(env, tdr0)
a0_list = acts0[0].tolist()
vis0 = [a for a in a0_list if a != 0]
assert sorted(vis0) == list(range(1, 51)), "CVRP-50 tour must cover all customers once"
rew0 = env.get_reward(tdr0, acts0)  # check_solution_validity runs inside (asserts)
xy0 = [tuple(map(float, td0["depot"][0].tolist()))] + [tuple(map(float, p)) for p in td0["locs"][0]]
demand01 = [0.0] + [float(x) for x in td0["demand"][0].tolist()]  # raw/40 per customer
L0, routes0 = raw_tour_len(a0_list, xy0, demand01, 1.0)
assert all(sum(demand01[c] for c in r) <= 1.0 + DE for r in routes0)
print(f"#0 scripted: {acts0.shape[-1]} actions, {len(routes0)} routes, "
      f"reward {float(rew0):.4f}, raw L {L0:.6f}, |diff| {abs(float(rew0) + L0):.2e}")
assert abs(float(rew0) + L0) < 1e-3

# --- 3. hand-built X td + scripted tour + raw recheck ------------------------
inst = load(str(X_FILE))
xy = inst["xy"]
gen_x = CVRPGenerator(num_loc=N_X)
env_x = CVRPEnv(generator=gen_x)
td_x = TensorDict(
    {
        "locs": torch.tensor(xy[1:] / 1000.0, dtype=torch.float32)[None],   # [1, 501, 2] on [0,1]
        "depot": torch.tensor(xy[0] / 1000.0, dtype=torch.float32)[None],   # [1, 2]
        "demand": torch.tensor(inst["demand"][1:], dtype=torch.float32)[None] / float(Q_X),
        "capacity": torch.tensor([[float(Q_X)]]),                          # raw Q, informational
    },
    batch_size=[1],
)
assert abs(float(td_x["demand"].max()) - 1.0 / Q_X) < 1e-6  # unit raw demands / Q
tdr_x = env_x.reset(td_x.clone())
assert tuple(tdr_x["locs"].shape) == (1, N_X + 1, 2)
assert torch.equal(tdr_x["locs"][0, 0], td_x["depot"][0])
acts_x = scripted_tour(env_x, tdr_x)
ax_list = acts_x[0].tolist()
vis_x = [a for a in ax_list if a != 0]
assert sorted(vis_x) == list(range(1, N_X + 1)), "X tour must cover all customers once"
rew_x = env_x.get_reward(tdr_x, acts_x)  # runs check_solution_validity (asserts)
demand01 = [0.0] + [float(d) / Q_X for d in inst["demand"].tolist()]
xy01 = [tuple(map(float, p)) for p in (xy / 1000.0)]
Lx01, routes_x = raw_tour_len(ax_list, xy01, demand01, 1.0)
diff01 = abs(float(rew_x) + Lx01)  # reward = -L on [0,1]; float32 td -> 4-decimal tolerance
print(f"X scripted: {acts_x.shape[-1]} actions, {len(routes_x)} routes, "
      f"reward {float(rew_x):.4f} vs raw L01 {Lx01:.6f} |diff| {diff01:.2e} (want < 5e-4)")
assert diff01 < 5e-4
assert all(sum(inst["demand"][c] for c in r) <= Q_X for r in routes_x)
print(f"X raw recheck OK: {len(routes_x)} routes, all <= Q={Q_X}")

# --- 4. GreedyEval with a fake policy (interface smoke) ----------------------


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, td, **kw):
        acts = scripted_tour(env_x, td)
        return {"actions": acts, "rewards": env_x.get_reward(td, acts)}


dl = DataLoader(TensorDictDataset(td_x), batch_size=1, collate_fn=TensorDictDataset.collate_fn)
ev = GreedyEval(env_x, progress=False)
t0 = time.time()
res = ev(_FakePolicy().cuda(), dl)
print("GreedyEval keys:", sorted(res.keys()))
assert tuple(res["rewards"].shape) == (1,)
assert res["actions"].shape[0] == 1 and res["actions"].shape[1] >= N_X
assert abs(float(res["rewards"][0]) - float(rew_x)) < 1e-3
print(f"GreedyEval OK ({time.time() - t0:.1f}s)")

# --- 5. run_pyvrp.py subprocess BKS cross-check ------------------------------
t0 = time.time()
r = subprocess.run(
    [sys.executable, str(REPO_ROOT / "src" / "run_pyvrp.py"), str(X_FILE), "2", "1"],
    capture_output=True, text=True, timeout=120,
)
wall = time.time() - t0
print("pyvrp subprocess rc", r.returncode, "stdout:", repr(r.stdout.strip()),
      "stderr_tail:", repr(r.stderr[-200:]))
assert r.returncode == 0
parts = r.stdout.strip().split(",")
assert parts[0] == "pyvrp" and len(parts) == 7
cost_v = float(parts[4])
assert parts[5] == "True"
assert abs(cost_v - BKS_X) / BKS_X < 0.01
print(f"pyvrp BKS cross-check OK: cost {cost_v} vs BKS {BKS_X} ({wall:.1f}s)")

print("ALL PROBE CHECKS PASSED")
