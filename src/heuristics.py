import numpy as np


def tour_length(tour, D):
    """Closed tour length on the [n, n] matrix D, O(n)."""
    n = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))


def nearest_neighbor(D, start=0):
    """Nearest-neighbor tour construction, O(n^2). Returns the tour as a
    permutation list of all n nodes, starting at `start`."""
    n = D.shape[0]
    unvisited = set(range(n)) - {start}
    tour = [start]
    while unvisited:
        cur = tour[-1]
        nxt = min(unvisited, key=lambda j: D[cur, j])
        unvisited.remove(nxt)
        tour.append(nxt)
    return tour


def two_opt(tour, D):
    """First-improvement 2-opt, expected O(n^2) (O(1) delta per (i, k) pair).
    Move (i, k): reverse tour[i:k+1].
    delta = D[a,c] + D[b,d] - D[a,b] - D[c,d], with a=tour[i-1], b=tour[i],
    c=tour[k], d=tour[(k+1)%n]. Repeats until no delta < -1e-9 remains.
    Mutates and returns tour."""
    n = len(tour)
    print(tour[0])
    # print(D)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            a = tour[i]
            b = tour[i+1]
            for k in range(i + 2, n):
                c = tour[k]
                d = tour[(k + 1) % n]
                if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -1e-9:
                    # print(f"{n} {i} {k}")
                    tour[i+1:k + 1] = tour[i+1:k + 1][::-1]
                    improved = True
                    break
                assert not improved
            if improved:
                break
        if not improved:
            break
    return tour


def or_opt(tour, D):
    """First-improvement OR-opt, expected O(n^2) (O(1) delta per move).
    Move (i, L, j): take the L = 1..3 consecutive nodes tour[i]..tour[i+L-1]
    (indices mod n) out of the tour and insert them after tour[j], where j is
    not within {i-1, ..., i+L-1} mod n.
    delta = D[a,d] - D[a,b] - D[c,d] - D[p,q] + D[p,b] + D[c,q], with
    a=tour[i-1], b=tour[i], c=tour[i+L-1], d=tour[i+L], p=tour[j], q=tour[j+1]
    (all indices mod n). Repeats until no delta < -1e-9 remains.
    Mutates and returns tour; the list may end cyclically rotated (same tour)."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for L in range(1, min(3, n - 2) + 1):
                a = tour[i - 1]
                b = tour[i]
                c = tour[(i + L - 1) % n]
                d = tour[(i + L) % n]
                base = D[a, d] - D[a, b] - D[c, d]
                for t in range(n - L - 1):
                    j = (i + L + t) % n
                    p = tour[j]
                    q = tour[(j + 1) % n]
                    if base - D[p, q] + D[p, b] + D[c, q] < -1e-9:
                        seg = [tour[(i + s) % n] for s in range(L)]
                        offset = (i + L) % n
                        rest = [tour[(offset + s) % n] for s in range(n - L)]
                        pos = (j - offset) % (n - L)
                        tour[:] = rest[:pos + 1] + seg + rest[pos + 1:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return tour


if __name__ == "__main__":
    import sys
    from gen import gen_tsp
    path = sys.argv[1] if len(sys.argv) > 1 else gen_tsp(n=100, seed=0)
    with open(path) as f:
        n = int(f.readline())
        D = np.loadtxt(f)
    tour = nearest_neighbor(D)
    nn = tour_length(tour, D)
    tour = two_opt(tour, D)
    l2 = tour_length(tour, D)
    tour = or_opt(tour, D)
    lor = tour_length(tour, D)
    print(f"n = {n}")
    print(f"NN     : {nn:.4f}")
    print(f"2-opt  : {l2:.4f}")
    print(f"or-opt : {lor:.4f}")
