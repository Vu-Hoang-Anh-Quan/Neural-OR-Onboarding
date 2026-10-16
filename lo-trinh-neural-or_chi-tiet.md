# Lộ trình nhập môn Neural OR — bản chi tiết 12 tuần

*Dự án [Research] ML-for-OR · Soạn 2026-10-11 · Dùng chung cho sinh viên mới. Khi giao: chép sang `07_students/<ho-ten>_<de-tai>/lo-trinh.md`, điền phần thông tin dưới đây và bảng theo dõi ở cuối.*

| Mục | Nội dung |
|---|---|
| Sinh viên | ……………… |
| Ngày bắt đầu tuần 1 | YYYY-MM-DD |
| Giảng viên hướng dẫn | ……………… |
| Repo Git | ……………… (code **không** đặt trong Google Drive) |
| Thời lượng | 12 tuần, khoảng 12–15 giờ/tuần |
| Đầu vào giả định | Biết Python, numpy; đại số tuyến tính và xác suất mức đại học. Chưa cần biết OR, PyTorch hay RL. |
| Phần cứng | Laptop cho tuần 1–7; từ tuần 8 cần GPU (máy lab hoặc Colab). Mọi kết quả phải ghi rõ CPU/GPU dùng. |

## Mục tiêu sau 12 tuần

1. Mô hình hoá và giải TSP/CVRP bằng MIP, heuristic tự viết và solver mạnh (OR-Tools, PyVRP, LKH-3); hiểu gap, best-known solution (BKS), đánh đổi chất lượng–thời gian.
2. Tự cài attention, GNN đơn giản và REINFORCE; huấn luyện được một policy giải TSP-20.
3. Tái lập Attention Model (NC02) và POMO (NC04) bằng RL4CO, **so công bằng** với solver OR, đo tổng quát hoá theo kích thước.

## Quy tắc làm việc (áp dụng từ tuần 1)

- **Luôn có baseline OR** và so **cùng ngân sách thời gian**. Ghi rõ phần cứng (ví dụ "PyVRP: 1 lõi CPU i7-12700; POMO: 1 GPU RTX 3090"). Không bao giờ chỉ so ML với ML.
- **Tái lập được:** mỗi lần chạy ghi một dòng vào `04_experiments/EXPERIMENT_LOG.md` (ngày, mã EXP, commit, config, seed, phần cứng, kết quả). Mã thí nghiệm: `EXP-ONB-<ten>-NNN`.
- **Đọc xong một paper → một file ghi chú** theo `01_literature/notes/_TEMPLATE_paper-note.md`, cập nhật cột "Trạng thái đọc" trong `literature-map.xlsx`.
- **Họp tuần 15–30 phút:** một bảng kết quả + một vướng mắc. Không trình bày slide lý thuyết.
- **Không tin con số của chính mình khi chưa kiểm tra khả thi**: mọi lời giải CVRP phải qua hàm `check()` (Phụ lục B).

---

## Tuần 0 — Chuẩn bị môi trường (2–3 giờ, trước tuần 1)

- [ ] Tạo repo `neural-or-onboarding` với cấu trúc: `data/`, `src/`, `notebooks/`, `results/`, `README.md`.
- [ ] Tạo môi trường Python ≥ 3.10:
  ```bash
  conda create -n nor python=3.11 -y && conda activate nor
  pip install numpy scipy pandas matplotlib jupyter vrplib pyvrp ortools
  ```
- [ ] Ghi `pip freeze > requirements.txt` vào repo; ghi phiên bản `pyvrp`, `ortools` vào README (thuật toán mặc định của PyVRP có thể thay đổi giữa các phiên bản).
- [ ] Tải LKH-3 (bản 3.0.9) tại http://webhotel4.ruc.dk/~keld/research/LKH-3/ và biên dịch: `tar xzf LKH-3.0.9.tgz && cd LKH-3.0.9 && make`.
- [ ] Tải bộ **X** của CVRPLIB (https://galgos.inf.puc-rio.br/cvrplib/), cả file `.vrp` và `.sol` (chứa BKS). Chọn **10 instance** trải đều số khách từ khoảng 100 đến khoảng 500 (ví dụ bắt đầu từ X-n101-k25); ghi danh sách vào `data/README.md`.
- [ ] Chép các file trong Phụ lục B vào `src/`, chạy thử `python src/run_pyvrp.py data/X-n101-k25.vrp 5 1`.

---

## Giai đoạn 1 — Nền OR (tuần 1–4)

### Tuần 1 — Phát biểu bài toán và mô hình MIP

**Mục tiêu:** viết được mô hình toán cho TSP/CVRP; thấy vì sao MIP không mở rộng được.

**Đọc**
- Toth & Vigo (eds.), *Vehicle Routing: Problems, Methods, and Applications*, 2nd ed., SIAM 2014 — chương 1 (các biến thể VRP).
- Công thức MTZ cho TSP (Miller–Tucker–Zemlin, 1960) — xem file `tsp_mtz.py` ở Phụ lục B.
- JSSP chỉ cần nắm định nghĩa (job, machine, operation, makespan) — chưa làm ở lộ trình này.

**Làm**
1. Viết vào `03_formulations/tsp-mtz_<ten>.md` (theo mẫu) mô hình:
   $$\min \sum_{i\neq j} c_{ij}x_{ij} \;\; \text{s.t.}\;\; \sum_{j} x_{ij}=1,\; \sum_{i} x_{ij}=1,\; u_i-u_j+(n-1)x_{ij}\le n-2 \;(i\neq j,\ i,j\ge 1),\; x_{ij}\in\{0,1\}.$$
2. Viết thêm mô hình CVRP hai chỉ số với biến tải $q_i$ thay cho $u_i$: $q_j \ge q_i + d_j - Q(1-x_{ij})$, $d_i \le q_i \le Q$, và ràng buộc số xe rời depot.
3. Chạy `tsp_mtz.py` với n = 8, 10, 12, 14, 16 (5 instance ngẫu nhiên mỗi n, seed 0–4, giới hạn 300 s). Ghi thời gian giải và trạng thái OPTIMAL hay không.

**Sản phẩm:** file formulation + bảng `n | thời gian TB | số instance giải tối ưu` + đồ thị thời gian theo n (trục log).

**Tự kiểm tra:** Ràng buộc MTZ loại subtour như thế nào? Vì sao thời gian tăng gần như hàm mũ? Relaxation LP của MTZ yếu nghĩa là gì?

### Tuần 2 — Heuristic xây dựng và 2-opt cho TSP

**Mục tiêu:** tự viết heuristic, hiểu local search và đánh giá delta O(1).

**Đọc**
- Johnson & McGeoch, "The Traveling Salesman Problem: A Case Study in Local Optimization", trong *Local Search in Combinatorial Optimization* (Aarts & Lenstra eds., 1997) — phần nearest neighbor, 2-opt, 3-opt, Or-opt.

**Làm**
1. Cài `nearest_neighbor(D, start=0)` và `two_opt(tour, D)` (khung ở Phụ lục C): first-improvement, tính delta của mỗi move bằng 4 cạnh, không tính lại toàn bộ tour.
2. Cài `or_opt` (dời đoạn 1–3 node) và chạy luân phiên 2-opt + Or-opt đến khi không cải thiện.
3. Chạy LKH-3 trên cùng instance (file `.par` mẫu ở Phụ lục B) làm tham chiếu.
4. Thí nghiệm: TSP ngẫu nhiên đều trong [0,1]², n = 20, 50, 100, 200, mỗi n 20 instance (seed cố định). Với n = 12 so thêm với MIP tối ưu.

**Sản phẩm:** bảng `n | NN gap | NN+2opt gap | NN+2opt+Oropt gap | thời gian` (gap so với LKH-3, hoặc so với tối ưu khi n=12).

**Tự kiểm tra:** Vì sao first-improvement và best-improvement cho kết quả khác nhau? Độ phức tạp một vòng 2-opt? Tại sao cần neighbor list khi n lớn?

### Tuần 3 — LNS cho CVRP

**Mục tiêu:** hiểu destroy/repair — nền tảng của nhiều phương pháp Neural OR sau này (NC03, NC58).

**Đọc**
- Shaw (1998), "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems", CP 1998 — ý tưởng LNS.
- Ropke & Pisinger (2006), "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows", *Transportation Science* 40(4) — mục toán tử removal/insertion và cơ chế adaptive.

**Làm**
1. Đọc instance bằng `load()` (Phụ lục B). Lời giải ban đầu: sweep hoặc nearest neighbor theo tải.
2. Destroy: `random_removal(k)`, `worst_removal(k)` (bỏ khách đóng góp cost lớn nhất), `related_removal(k)` (Shaw: khách gần nhau). k ngẫu nhiên trong 10–30% số khách.
3. Repair: `greedy_insertion` (chèn vào vị trí rẻ nhất còn khả thi tải); thêm `regret-2 insertion` nếu kịp.
4. Chấp nhận: simulated annealing (nhiệt độ giảm tuyến tính theo thời gian). Dừng theo thời gian.
5. Mỗi lời giải cuối phải qua `check()`.

**Sản phẩm:** `src/lns.py` + bảng trên 10 instance X, 60 s/instance, 3 seed: `instance | BKS | cost TB ± std | gap %`.

**Tự kiểm tra:** Toán tử nào cải thiện nhiều nhất? Vì sao cần chấp nhận lời giải xấu hơn? Bỏ quá nhiều/quá ít khách thì sao?

### Tuần 4 — So sánh solver và đánh đổi chất lượng–thời gian

**Mục tiêu:** biết "lời giải tốt" của CVRP trông thế nào; đây là thước đo cho mọi mô hình ML về sau.

**Đọc**
- Vidal (2022), "Hybrid Genetic Search for the CVRP: Open-Source Implementation and SWAP* Neighborhood", *Computers & OR* — arXiv 2012.10384.
- Wouda, Lan & Kool (2024), "PyVRP: A High-Performance VRP Solver Package", *INFORMS Journal on Computing* — DOI 10.1287/ijoc.2023.0055 (arXiv 2403.13795).
- Uchoa et al. (2017), "New Benchmark Instances for the Capacitated Vehicle Routing Problem", *EJOR* — cách sinh bộ X.
- OR-Tools Routing: https://developers.google.com/optimization/routing

**Làm**
1. Chạy `run_pyvrp.py`, `run_ortools.py` và `lns.py` trên 10 instance × thời gian {1, 10, 60} s × 3 seed (OR-Tools: 1 lần mỗi mức thời gian).
2. Ghi CSV `results/week4.csv`: `solver, instance, time_limit, seed, cost, feasible, wallclock, hardware`.
3. Vẽ đồ thị gap (%) theo thời gian (trục log) cho từng solver.

**Sản phẩm:** CSV + đồ thị + nhận xét 1 trang.

**Tự kiểm tra (mốc 1, họp với GVHD):** Gap của LNS tự viết so với PyVRP ở 60 s? Vì sao? BKS là gì và vì sao không gọi là "tối ưu"? Nếu một mô hình ML đạt gap 2% trong 0.1 s/instance thì có tốt không — cần thêm thông tin gì?

---

## Giai đoạn 2 — Nền ML (tuần 5–8)

### Tuần 5 — PyTorch và attention

**Đọc**
- *Dive into Deep Learning* (https://d2l.ai) — chương Attention Mechanisms and Transformers.
- *The Annotated Transformer* (https://nlp.seas.harvard.edu/annotated-transformer/).

**Làm**
1. `pip install torch` (bản phù hợp CUDA của máy). Ôn training loop: dataset, DataLoader, optimizer, `loss.backward()`.
2. Tự cài `MultiHeadAttention` (Q, K, V, chia head, scaled dot-product, mask). Kiểm tra: chép trọng số sang `torch.nn.MultiheadAttention`, đầu ra phải `torch.allclose(..., atol=1e-5)`.
3. Cài một Transformer encoder 3 lớp **không positional encoding** cho tập điểm 2D; giải thích vì sao TSP cần bất biến hoán vị.

**Sản phẩm:** `src/attention.py` + test allclose chạy qua.

### Tuần 6 — Graph Neural Network

**Đọc**
- Stanford CS224W (https://web.stanford.edu/class/cs224w/) — các bài giảng về GNN, message passing, GCN/GAT.
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io — "Introduction by Example".

**Làm**
1. GCN phân lớp node trên Cora (`torch_geometric.datasets.Planetoid`). Ghi accuracy test.
2. **Bài cầu nối:** với 1.000 TSP-20 ngẫu nhiên, lấy tour LKH-3 làm nhãn; xây đồ thị k-NN (k=8); huấn luyện GNN phân loại cạnh "có trong tour hay không". Báo precision/recall và tỷ lệ cạnh tour nằm trong top-k. Đây chính là ý tưởng "heatmap" của NC20, NC10.

**Sản phẩm:** notebook + bảng kết quả. Ghi chú: một heatmap k-NN không học cho kết quả thế nào so với GNN? (liên hệ NC44, NC66).

### Tuần 7 — Policy gradient

**Đọc**
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (bản PDF tại http://incompleteideas.net/book/the-book-2nd.html) — chương 3 (MDP) và chương 13 (Policy Gradient Methods).
- OpenAI Spinning Up (https://spinningup.openai.com) — Introduction to RL phần 1–3, Vanilla Policy Gradient.

**Làm**
1. `pip install gymnasium`. Cài REINFORCE cho `CartPole-v1`: (a) không baseline, (b) baseline trung bình động. Mỗi biến thể 5 seed.
2. Vẽ đường cong return theo episode (mean ± std qua seed).

**Sản phẩm:** `src/reinforce_cartpole.py` + đồ thị.

**Tự kiểm tra:** Vì sao baseline không làm lệch gradient? Vì sao phương sai cao là vấn đề chính của REINFORCE?

### Tuần 8 — REINFORCE giải TSP-20

**Mục tiêu:** ghép attention + REINFORCE thành một neural solver nhỏ (phiên bản thu gọn của NC02).

**Làm** (khung ở Phụ lục C)
1. Encoder: Transformer 3 lớp từ tuần 5. Decoder: tại mỗi bước, attention từ ngữ cảnh (node đầu, node hiện tại, trung bình graph) tới các node chưa thăm; mask node đã thăm.
2. Huấn luyện trên TSP-20 sinh ngẫu nhiên mỗi batch (batch 512), Adam lr 1e-4, 100 epoch × 100 batch (điều chỉnh theo GPU, **ghi thời gian huấn luyện**).
3. Hai baseline: (a) trung bình động; (b) greedy rollout của bản sao policy, cập nhật khi tốt hơn có ý nghĩa (như NC02).
4. Đánh giá trên **1.000 instance cố định** (seed 1234): greedy decoding và sampling 128 lời giải; so với NN, NN+2opt (tuần 2), LKH-3.

**Sản phẩm:** bảng `phương pháp | gap TB so với LKH-3 | thời gian tổng trên 1.000 instance | phần cứng` + đường cong huấn luyện.

**Mốc 2 (họp GVHD):** sinh viên giải thích được toàn bộ luồng huấn luyện; kết quả greedy rollout tốt hơn trung bình động hay không; vì sao vẫn có thể thua 2-opt.

---

## Giai đoạn 3 — Tái lập paper và đánh giá công bằng (tuần 9–12)

### Tuần 9 — Đọc kỹ NC02 và NC04

**Đọc:** Kool et al., *Attention, Learn to Solve Routing Problems!* (ICLR 2019) [NC02]; Kwon et al., *POMO* (NeurIPS 2020) [NC04]. Viết 2 file ghi chú theo mẫu.

**Trả lời trong ghi chú:**
- NC02: vai trò của greedy rollout baseline; vì sao decoder dùng context gồm node đầu và node hiện tại; cách xử lý ràng buộc tải trong CVRP.
- NC04: multiple starting nodes và shared baseline thay rollout baseline thế nào; instance augmentation ×8 là gì; vì sao POMO huấn luyện nhanh hơn AM.
- Cả hai: so với baseline OR nào, cùng thời gian hay không, phần cứng ra sao.

### Tuần 10 — Chạy AM và POMO với RL4CO

**Đọc:** Berto et al., *RL4CO* (KDD 2025) [NC17]; quickstart: https://rl4co.readthedocs.io/en/latest/examples/1-quickstart/

**Làm**
1. `pip install rl4co`; chạy notebook quickstart, ghi phiên bản.
2. Huấn luyện AM và POMO cho TSP50 và CVRP50 với cấu hình mặc định của thư viện; giảm số epoch nếu thiếu GPU nhưng **ghi rõ** số epoch, thời gian huấn luyện, GPU.
3. Lưu checkpoint và config vào repo (checkpoint lớn để ngoài Drive).

**Sản phẩm:** 4 checkpoint + log huấn luyện + 4 dòng EXPERIMENT_LOG.

### Tuần 11 — Đánh giá công bằng

**Đọc:** Xia et al., *Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale TSP* (ICML 2024) [NC44]; Pan et al. (ICLR 2026) [NC66] — chỉ phần giao thức đánh giá.

**Giao thức bắt buộc**
1. Tập test: 1.000 CVRP50 cố định (seed 1234, sinh theo phân phối của RL4CO) và 10 instance X từ tuần 0.
2. POMO ba chế độ: greedy, greedy + augmentation ×8, sampling. Ghi **tổng thời gian trên cả tập** và thời gian/instance, kèm GPU.
3. PyVRP và OR-Tools với ngân sách thời gian/instance bằng (a) thời gian/instance của POMO, (b) 1 s, (c) 10 s. Ghi CPU và số lõi.
4. 3 seed cho mọi phương pháp có ngẫu nhiên; báo mean ± std.
5. Bảng chính: `phương pháp | gap so với BKS (X) hoặc PyVRP 60 s (CVRP50) | thời gian/instance | phần cứng`.

**Sản phẩm:** bảng chính + đồ thị gap–thời gian gộp cả ML và OR trên cùng trục.

**Tự kiểm tra:** Ở ngân sách thời gian nào POMO có lợi thế? Lợi thế đó còn không nếu đếm cả thời gian huấn luyện? So GPU với 1 lõi CPU có công bằng không, cần ghi chú gì?

### Tuần 12 — Tổng quát hoá theo kích thước và báo cáo tái lập

**Đọc:** lướt abstract và hình chính của NC12 (LEHD), NC39 (INViT), NC38 (GLOP) để biết các cách khắc phục.

**Làm**
1. Dùng POMO huấn luyện ở n=50, test trên CVRP ngẫu nhiên n = 100, 200, 500 (mỗi n 100 instance) và trên 10 instance X.
2. Vẽ gap theo n cho POMO và PyVRP (cùng thời gian).
3. Viết **báo cáo tái lập 3–5 trang** (`04_experiments/results/EXP-ONB-<ten>-report.md`): thiết lập, bảng, đồ thị, nhận xét trung thực (kể cả chỗ không tái lập được số của paper).

**Mốc 3 — quyết định tiếp tục (họp GVHD, dùng bảng chấm ở Phụ lục D).**

## Phụ lục A — Lịch đọc tóm tắt

| Tuần | Đọc | Loại |
|---|---|---|
| 1 | Toth & Vigo (2014) ch. 1; MTZ | Sách |
| 2 | Johnson & McGeoch (1997) | Chương sách |
| 3 | Shaw (1998); Ropke & Pisinger (2006) | Bài báo |
| 4 | Vidal (2022) HGS; Wouda et al. (2024) PyVRP; Uchoa et al. (2017) | Bài báo |
| 5 | d2l.ai Attention; Annotated Transformer | Online |
| 6 | CS224W (GNN); PyG intro | Online |
| 7 | Sutton & Barto ch. 3, 13; Spinning Up | Sách/online |
| 9 | NC02, NC04 | Bài báo |
| 10 | NC17 + RL4CO quickstart | Bài báo/online |
| 11 | NC44, NC66 | Bài báo |
| 12 | NC12, NC39, NC38 (lướt) | Bài báo |

## Phụ lục B — Mã khởi đầu (đã chạy thử với vrplib 2.x, PyVRP 0.14, OR-Tools 9.15)

`src/cvrp_utils.py`
```python
"""Tiện ích CVRP: đọc instance, ma trận khoảng cách, cost, kiểm tra khả thi.
Quy ước: node 0 là depot; lời giải là list route, mỗi route là list khách (không chứa depot)."""
import numpy as np, vrplib

def load(path):
    inst = vrplib.read_instance(path)                      # tự tính edge_weight cho EUC_2D
    D = np.floor(inst["edge_weight"] + 0.5).astype(int)    # quy ước nint của CVRPLIB
    return {"name": inst["name"], "D": D, "demand": np.asarray(inst["demand"]),
            "Q": int(inst["capacity"]), "xy": np.asarray(inst["node_coord"])}

def route_cost(r, D):
    return int(D[0, r[0]] + sum(D[a, b] for a, b in zip(r, r[1:])) + D[r[-1], 0]) if r else 0

def cost(routes, D):
    return sum(route_cost(r, D) for r in routes)

def check(routes, inst):
    """(khả_thi, thông_báo): mỗi khách đúng một lần, không route nào vượt tải."""
    n = len(inst["demand"]) - 1
    if sorted(c for r in routes for c in r) != list(range(1, n + 1)):
        return False, "thiếu hoặc lặp khách hàng"
    for r in routes:
        if inst["demand"][r].sum() > inst["Q"]:
            return False, f"route vượt tải: {inst['demand'][r].sum()} > {inst['Q']}"
    return True, "ok"

def gap(c, ref):
    return 100.0 * (c - ref) / ref
```

`src/gen.py` — sinh CVRP ngẫu nhiên định dạng VRPLIB
```python
import numpy as np, vrplib

def gen_cvrp(n=100, capacity=50, seed=0, path=None):
    rng = np.random.default_rng(seed)
    xy = rng.integers(0, 1001, size=(n + 1, 2))          # node 0 = depot
    demand = rng.integers(1, 10, size=n + 1); demand[0] = 0
    path = path or f"rand-n{n}-s{seed}.vrp"
    vrplib.write_instance(path, {
        "NAME": path[:-4], "TYPE": "CVRP", "DIMENSION": n + 1, "CAPACITY": capacity,
        "EDGE_WEIGHT_TYPE": "EUC_2D", "NODE_COORD_SECTION": xy,
        "DEMAND_SECTION": demand, "DEPOT_SECTION": np.array([1, -1])})
    return path
```

`src/run_pyvrp.py` — dùng: `python run_pyvrp.py <file.vrp> <giây> <seed>`
```python
import sys, time
from pyvrp import read, solve
from pyvrp.stop import MaxRuntime

path, sec, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
data = read(path, round_func="round")                     # cùng quy ước làm tròn với CVRPLIB
t = time.perf_counter()
res = solve(data, stop=MaxRuntime(sec), seed=seed)
print(f"pyvrp,{path},{sec},{seed},{res.cost()},{res.is_feasible()},{time.perf_counter()-t:.1f}")
```

`src/run_ortools.py` — dùng: `python run_ortools.py <file.vrp> <giây>`
```python
import sys, time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from cvrp_utils import load, cost, check

path, sec = sys.argv[1], int(sys.argv[2])
inst = load(path); D, dem, Q = inst["D"], inst["demand"], inst["Q"]
n_veh = len(dem) - 1
mgr = pywrapcp.RoutingIndexManager(len(dem), n_veh, 0)
rm = pywrapcp.RoutingModel(mgr)
dist_cb = rm.RegisterTransitCallback(lambda i, j: int(D[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
rm.SetArcCostEvaluatorOfAllVehicles(dist_cb)
dem_cb = rm.RegisterUnaryTransitCallback(lambda i: int(dem[mgr.IndexToNode(i)]))
rm.AddDimensionWithVehicleCapacity(dem_cb, 0, [Q] * n_veh, True, "Capacity")
p = pywrapcp.DefaultRoutingSearchParameters()
p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
p.time_limit.seconds = sec
t = time.perf_counter(); sol = rm.SolveWithParameters(p)
routes = []
for v in range(n_veh):
    i, r = rm.Start(v), []
    while not rm.IsEnd(i):
        node = mgr.IndexToNode(i)
        if node: r.append(node)
        i = sol.Value(rm.NextVar(i))
    if r: routes.append(r)
print(f"ortools,{path},{sec},-,{cost(routes, D)},{check(routes, inst)[0]},{time.perf_counter()-t:.1f}")
```

`src/tsp_mtz.py` — TSP bằng MIP (SCIP qua OR-Tools), chỉ cho n ≤ 16
```python
import numpy as np
from ortools.linear_solver import pywraplp

def solve_tsp_mtz(D, time_limit_s=60):
    n = len(D)
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): s.BoolVar(f"x_{i}_{j}") for i in range(n) for j in range(n) if i != j}
    u = {i: s.NumVar(1, n - 1, f"u_{i}") for i in range(1, n)}
    for i in range(n):
        s.Add(sum(x[i, j] for j in range(n) if j != i) == 1)
        s.Add(sum(x[j, i] for j in range(n) if j != i) == 1)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s.Add(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2)
    s.Minimize(sum(D[i][j] * x[i, j] for (i, j) in x))
    s.SetTimeLimit(time_limit_s * 1000)
    status = s.Solve()
    tour, cur = [0], 0
    for _ in range(n - 1):
        cur = next(j for j in range(n) if j != cur and x[cur, j].solution_value() > 0.5)
        tour.append(cur)
    return s.Objective().Value(), tour, status == pywraplp.Solver.OPTIMAL

if __name__ == "__main__":
    rng = np.random.default_rng(0); xy = rng.random((10, 2))
    print(solve_tsp_mtz(np.linalg.norm(xy[:, None] - xy[None], axis=-1)))
```

File tham số LKH-3 cho TSP (`tsp.par`):
```
PROBLEM_FILE = tsp.tsp
OUTPUT_TOUR_FILE = tsp.tour
RUNS = 1
MAX_TRIALS = 1000
SEED = 1
```
Chạy: `./LKH tsp.par`. Toạ độ trong [0,1] cần nhân với hệ số lớn (ví dụ 10⁶) và làm tròn khi ghi file `.tsp`, vì LKH dùng khoảng cách nguyên.

## Phụ lục C — Khung code sinh viên tự hoàn thiện

`two_opt` (tuần 2)
```python
def tour_length(tour, D): ...
def nearest_neighbor(D, start=0): ...          # trả về list hoán vị

def two_opt(tour, D):
    """First-improvement 2-opt. Move (i, k): đảo đoạn tour[i:k+1].
    delta = D[a,c] + D[b,d] - D[a,b] - D[c,d], với a=tour[i-1], b=tour[i], c=tour[k], d=tour[(k+1)%n].
    Lặp đến khi không còn delta < -1e-9."""
    ...
```

`lns` (tuần 3)
```python
def lns(inst, time_limit, seed, destroy_ops, repair_op, T0=..., frac=(0.1, 0.3)):
    """Trả về routes tốt nhất. Mỗi vòng: chọn destroy ngẫu nhiên, bỏ k khách,
    repair, chấp nhận theo simulated annealing, cập nhật best nếu khả thi và tốt hơn."""
    ...
```

REINFORCE cho TSP-20 (tuần 8)
```python
class Encoder(nn.Module): ...                 # Transformer 3 lớp từ tuần 5, input (B, n, 2)
class Decoder(nn.Module):
    def forward(self, h, mask, first, last):  # trả log-prob trên các node (B, n), node đã thăm bị mask
        ...

def rollout(policy, xy, greedy=False):
    """Sinh tour cho cả batch; trả (tour (B, n), tổng log-prob (B,), độ dài tour (B,))."""
    ...

for epoch in range(E):
    for _ in range(steps):
        xy = torch.rand(B, 20, 2, device=dev)
        tour, logp, L = rollout(policy, xy)
        with torch.no_grad():
            b = baseline(xy)                  # trung bình động hoặc greedy rollout của bản sao policy
        loss = ((L - b) * logp).mean()        # cực tiểu độ dài tour
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    # cuối epoch: đánh giá trên tập validation cố định; cập nhật baseline nếu dùng greedy rollout
```

## Phụ lục D — Bảng chấm các mốc

| Tiêu chí | Mốc 1 (tuần 4) | Mốc 2 (tuần 8) | Mốc 3 (tuần 12) |
|---|---|---|---|
| Hoàn thành sản phẩm | Formulation, 2-opt, LNS, bảng solver | Attention, GNN, REINFORCE TSP-20 | AM/POMO, đánh giá công bằng, báo cáo |
| Đúng đắn | Lời giải qua `check()`; gap tính đúng theo BKS | Test allclose qua; đường cong huấn luyện hợp lý | Giao thức tuần 11 đủ 5 điểm |
| Tái lập | Seed, phiên bản, phần cứng trong log | Như mốc 1 + thời gian huấn luyện | Người khác chạy lại được từ README |
| Hiểu biết | Trả lời câu hỏi tự kiểm tra tuần 4 | Giải thích luồng REINFORCE | Nêu được khi nào NCO có lợi, khi nào không |
| Tự lập | Mức độ cần hỗ trợ (1–5) | (1–5) | (1–5) |

**Quyết định ở mốc 3:** tiếp tục theo hướng Neural OR / chuyển hướng thiên OR cổ điển hoặc thiên ML / dừng.

## Theo dõi tiến độ

| Tuần | Việc đã làm | Kết quả chính | Vướng mắc | Nhận xét GVHD |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 (mốc 1) | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 (mốc 2) | | | | |
| 9 | | | | |
| 10 | | | | |
| 11 | | | | |
| 12 (mốc 3) | | | | |
