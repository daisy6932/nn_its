# Archive_new/plotting.py
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


def save_path_metrics_csv(metrics, path):
    if not metrics:
        return
    keys = list(metrics[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for m in metrics:
            w.writerow(m)


def _cluster_ids_by_round(Z: np.ndarray, round_decimals: int) -> np.ndarray:
    """Cluster rows by rounding then exact match."""
    Zr = np.round(Z, round_decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)
    return cid


def _cluster_ids_by_eps(Z: np.ndarray, snap_eps: float) -> np.ndarray:
    """
    Cluster rows by threshold: rows i,j are in same cluster if max|Zi-Zj| <= snap_eps.
    Greedy union-find style (O(K^2)), fine for K~10-50.
    """
    K = Z.shape[0]
    cid = -np.ones(K, dtype=int)
    cur = 0
    for i in range(K):
        if cid[i] != -1:
            continue
        cid[i] = cur
        for j in range(i + 1, K):
            if cid[j] != -1:
                continue
            if np.max(np.abs(Z[i] - Z[j])) <= snap_eps:
                cid[j] = cur
        cur += 1
    return cid


def plot_scaf_dendrogram(
    lambda_values,
    all_Z_list,
    round_decimals: int = 2,
    title: str = "",
    save_path: str = "dendrogram.png",
    snap_eps: float | None = None,
    labels=None,
    xlabel: str = "Treatment Index",
):
    """
    Build a SCAF-style dendrogram from the solution path.

    Parameters
    ----------
    lambda_values : array-like, shape (L,)
    all_Z_list    : list of np.ndarray, each shape (K, r)
    round_decimals: used when snap_eps is None
    snap_eps      : if provided, cluster rows by max|Zi-Zj| <= snap_eps (recommended)
    labels        : optional labels for leaves (length K)
    xlabel        : default 'Treatment Index' (was 'Class Index')
    """
    if not all_Z_list:
        return None

    L = len(all_Z_list)
    K = all_Z_list[0].shape[0]

    # build clusterings over lambda
    all_clusterings = []
    for Z in all_Z_list:
        if Z.shape[0] != K:
            raise ValueError("Inconsistent K across all_Z_list.")
        if snap_eps is None:
            cid = _cluster_ids_by_round(Z, round_decimals)
        else:
            cid = _cluster_ids_by_eps(Z, float(snap_eps))
        all_clusterings.append(cid)

    # linkage construction by tracking merges along the path
    linkage_list = []
    cluster_members = {i: frozenset([i]) for i in range(K)}
    linkage_ids = {i: i for i in range(K)}
    cluster_heights = {i: 0.0 for i in range(K)}
    next_id = K
    merges_found = False

    for t in range(1, L):
        lam = float(lambda_values[t])
        groups = all_clusterings[t]

        merger_map = {}
        active = list(cluster_members.keys())
        for cid in active:
            item = next(iter(cluster_members[cid]))
            new_gid = int(groups[item])
            merger_map.setdefault(new_gid, set()).add(cid)

        for _, to_merge in merger_map.items():
            if len(to_merge) <= 1:
                continue
            merges_found = True

            lst = sorted(list(to_merge), key=lambda x: cluster_heights[x])
            while len(lst) > 1:
                a = lst.pop(0)
                b = lst.pop(0)
                la = linkage_ids[a]
                lb = linkage_ids[b]

                new = next_id
                new_h = lam
                new_mem = cluster_members[a].union(cluster_members[b])

                linkage_list.append([la, lb, new_h, len(new_mem)])

                # update bookkeeping
                del cluster_members[a]
                del cluster_members[b]
                cluster_members[new] = new_mem
                linkage_ids[new] = new
                cluster_heights[new] = new_h
                next_id += 1

                lst.append(new)
                lst.sort(key=lambda x: cluster_heights[x])

    if (not linkage_list) or (not merges_found):
        return None

    Z_link = np.array(linkage_list, dtype=float)

    # default labels
    if labels is None:
        labels = np.arange(K)
    else:
        if len(labels) != K:
            raise ValueError("labels length must equal K.")

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"$\lambda$ (fusion strength)")
    sch.dendrogram(Z_link, labels=labels, leaf_rotation=0.0, leaf_font_size=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
