# Lộ trình xây dựng nền tảng Neural OR — <Họ tên sinh viên>

*Sao chép vào `07_students/<ho-ten>_<de-tai>/lo-trinh.md` khi giao. Bản đầy đủ (code mẫu, giao thức đánh giá, bảng chấm): `lo-trinh-neural-or_chi-tiet.md`.*

- **Ngày bắt đầu:** YYYY-MM-DD · **GVHD:** ……… · **Repo Git:** ………
- **Thời lượng:** 12 tuần, khoảng 12–15 giờ/tuần
- **Đầu vào:** Python, numpy; đại số tuyến tính, xác suất. Chưa cần OR, PyTorch, RL.
- **Phần cứng:** laptop cho giai đoạn 1–2; GPU (lab hoặc Colab) từ tuần 8.

## Nguyên tắc

- **OR trước, ML sau:** biết lời giải tốt trông thế nào trước khi huấn luyện mô hình sinh lời giải.
- **Luôn so với baseline OR** (OR-Tools, PyVRP, LKH-3) **cùng ngân sách thời gian**, ghi rõ phần cứng.
- **Tái lập được:** seed, phiên bản thư viện, phần cứng, commit hash ghi vào `04_experiments/EXPERIMENT_LOG.md`.
- **Đọc paper → ghi chú** theo `01_literature/notes/_TEMPLATE_paper-note.md`.
- **Họp tuần:** một bảng kết quả + một vướng mắc.

## Giai đoạn 1 — Nền OR (tuần 1–4)

**Mục tiêu:** mô hình hoá và giải TSP/CVRP; hiểu gap, best-known solution (BKS), đánh đổi chất lượng–thời gian.

| Tuần | Đọc | Làm | Sản phẩm |
|---|---|---|---|
| 1 | Toth & Vigo, *Vehicle Routing* (SIAM 2014), ch. 1; công thức MTZ | Viết MIP cho TSP (MTZ) và CVRP; giải TSP n = 8–16 bằng OR-Tools/SCIP | File trong `03_formulations/`; bảng thời gian giải theo n |
| 2 | Johnson & McGeoch (1997), TSP local search | Tự code nearest neighbor, 2-opt, Or-opt; so với LKH-3 | Bảng gap theo n = 20–200 |
| 3 | Shaw (1998) LNS; Ropke & Pisinger (2006) ALNS | Tự code LNS cho CVRP (random/worst/related removal + greedy insertion) | Bảng gap trên 10 instance CVRPLIB (bộ X) |
| 4 | Vidal (2022) HGS; Wouda et al. (2024) PyVRP; Uchoa et al. (2017) | Chạy OR-Tools, PyVRP, LNS tự viết với 1/10/60 s, 3 seed | Đồ thị gap theo thời gian |

**Kiểm tra cuối giai đoạn:**
- [ ] Giải thích được BKS, gap, vì sao MIP không mở rộng được.
- [ ] Giải thích được vì sao LNS tự viết kém PyVRP.

## Giai đoạn 2 — Nền ML (tuần 5–8)

**Mục tiêu:** đủ PyTorch, attention, GNN và policy gradient để đọc và chạy các mô hình NCO.

| Tuần | Đọc | Làm | Sản phẩm |
|---|---|---|---|
| 5 | d2l.ai — Attention & Transformers; *The Annotated Transformer* | Tự cài multi-head attention, kiểm tra khớp `torch.nn.MultiheadAttention` | Code + test |
| 6 | CS224W (phần GNN); PyTorch Geometric intro | GCN trên Cora; GNN dự đoán cạnh thuộc tour TSP-20 (nhãn từ LKH-3) | Notebook + bảng kết quả |
| 7 | Sutton & Barto ch. 3, 13; Spinning Up (VPG) | REINFORCE cho CartPole, có/không baseline, 5 seed | Đồ thị return |
| 8 | — | Attention + REINFORCE giải TSP-20; so với NN, 2-opt, LKH-3 trên 1.000 instance cố định | Bảng gap + thời gian + phần cứng |

**Kiểm tra cuối giai đoạn:**
- [ ] Giải thích được vì sao cần baseline trong REINFORCE và vì sao TSP cần mô hình bất biến hoán vị.
- [ ] Mô hình TSP-20 huấn luyện ổn định; trả lời được vì sao vẫn có thể thua 2-opt.

## Giai đoạn 3 — Nền NCO: tái lập và đánh giá (tuần 9–12)

**Mục tiêu:** hiểu và chạy lại các mô hình nền, biết đánh giá công bằng và nhận ra giới hạn của NCO.

| Tuần | Đọc | Làm | Sản phẩm |
|---|---|---|---|
| 9 | NC02 (Attention Model), NC04 (POMO) | Ghi chú: baseline, decoder, augmentation, so với OR thế nào | 2 file ghi chú |
| 10 | NC17 (RL4CO) + quickstart | Huấn luyện AM, POMO cho TSP50, CVRP50; ghi thời gian huấn luyện | Checkpoint + log |
| 11 | NC44, NC66 (đánh giá phản biện) | So POMO với PyVRP, OR-Tools cùng thời gian/instance, 3 seed | Bảng + đồ thị gap–thời gian |
| 12 | Lướt NC12, NC38, NC39 (quy mô lớn) | Test POMO (huấn luyện n = 50) ở n = 100, 200, 500 và CVRPLIB | Báo cáo tái lập 3–5 trang |

**Kiểm tra cuối giai đoạn:**
- [ ] Báo cáo tái lập chạy lại được từ README.
- [ ] Nêu được khi nào NCO có lợi thế so với solver OR, khi nào không.

## Tài nguyên

- **Solver:** OR-Tools (https://developers.google.com/optimization/routing) · PyVRP (https://pyvrp.org) · LKH-3 (http://webhotel4.ruc.dk/~keld/research/LKH-3/)
- **Dữ liệu:** CVRPLIB (https://galgos.inf.puc-rio.br/cvrplib/) · đọc file bằng `vrplib`
- **ML:** https://d2l.ai · https://nlp.seas.harvard.edu/annotated-transformer/ · https://web.stanford.edu/class/cs224w/ · https://pytorch-geometric.readthedocs.io · http://incompleteideas.net/book/the-book-2nd.html · https://spinningup.openai.com
- **NCO:** RL4CO (https://rl4co.readthedocs.io) · tổng quan nội bộ `01_literature/literature-review_neural-co.md`

## Theo dõi tiến độ

| Tuần | Việc đã làm | Kết quả chính | Vướng mắc | Nhận xét GVHD |
|---|---|---|---|---|
| 1 | | | | |
| … | | | | |
| 12 | | | | |
