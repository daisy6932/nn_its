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


def save_true_group_heatmap(groups, n_treatments, save_path, title="True Group Structure"):
    """
    groups: e.g. [[0,1,2,3,4],[5,6,7,8,9]]
    Save a KxK matrix where entry (i,j)=1 if i,j are in same true group, else 0.
    """
    M = np.zeros((n_treatments, n_treatments), dtype=float)

    for grp in groups:
        for i in grp:
            for j in grp:
                M[i, j] = 1.0

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(n_treatments))
    plt.yticks(range(n_treatments))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def _cluster_ids_by_round(Z: np.ndarray, round_decimals: int) -> np.ndarray:
    Zr = np.round(Z, round_decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)
    return cid


def _cluster_ids_by_eps(Z: np.ndarray, snap_eps: float) -> np.ndarray:
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
    if not all_Z_list:
        return None

    L = len(all_Z_list)
    K = all_Z_list[0].shape[0]

    all_clusterings = []
    for Z in all_Z_list:
        if Z.shape[0] != K:
            raise ValueError("Inconsistent K across all_Z_list.")
        if snap_eps is None:
            cid = _cluster_ids_by_round(Z, round_decimals)
        else:
            cid = _cluster_ids_by_eps(Z, float(snap_eps))
        all_clusterings.append(cid)

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
