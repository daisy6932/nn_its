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

def save_selected_Z_heatmap(Z, save_path, title="Selected Z* pairwise distance"):
    """
    Z: (K, d) selected fused parameter matrix
    Save pairwise row distance heatmap.
    """
    Z = np.asarray(Z)
    K = Z.shape[0]
    D = np.zeros((K, K), dtype=float)

    for i in range(K):
        for j in range(K):
            D[i, j] = np.linalg.norm(Z[i] - Z[j])

    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(K))
    plt.yticks(range(K))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_cycle_summary(cycle_records, save_path):
    """
    cycle_records: list of dicts, each containing:
      - cycle
      - cycle_val_mse
      - cycle_score
      - stability_to_prev
    """
    if not cycle_records:
        return None

    cycles = [r["cycle"] for r in cycle_records]
    val_mse = [r.get("cycle_val_mse", np.nan) for r in cycle_records]
    cycle_score = [r.get("cycle_score", np.nan) for r in cycle_records]
    stability = [r.get("stability_to_prev", np.nan) for r in cycle_records]

    plt.figure(figsize=(8, 5))
    plt.plot(cycles, val_mse, marker="o", label="Validation MSE")
    plt.plot(cycles, cycle_score, marker="s", label="Cycle Score")
    plt.plot(cycles, stability, marker="^", label="Stability to Prev")
    plt.xlabel("Cycle")
    plt.ylabel("Value")
    plt.title("Cycle-level Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_lambda_selection_curve(metrics, picked_lambda_index, save_path):
    """
    metrics: list of dicts from one cycle
    """
    if not metrics:
        return None

    lambdas = [m["lambda"] for m in metrics]
    reg_loss = [m.get("reg_loss", np.nan) for m in metrics]
    val_mse = [m.get("val_mse", np.nan) for m in metrics]
    c3 = [m.get("clusters_3d", np.nan) for m in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, reg_loss, marker="o", label="Reg Loss")
    if not all(np.isnan(val_mse)):
        plt.plot(lambdas, val_mse, marker="s", label="Val MSE")
    plt.plot(lambdas, c3, marker="^", label="Clusters_3d")

    if picked_lambda_index is not None and 0 <= picked_lambda_index < len(lambdas):
        x = lambdas[picked_lambda_index]
        plt.axvline(x=x, linestyle="--", label=f"Picked λ={x:.4f}")

    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("Lambda")
    plt.ylabel("Metric")
    plt.title("Lambda Selection Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def save_selected_cluster_heatmap(Z, save_path, round_decimals=3, title="Selected Cluster Structure"):
    """
    Convert selected Z into cluster ids and save same-cluster matrix.
    """
    Z = np.asarray(Z)
    Zr = np.round(Z, round_decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)

    K = len(cid)
    M = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            M[i, j] = 1.0 if cid[i] == cid[j] else 0.0

    plt.figure(figsize=(6, 5))
    plt.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Treatment Index")
    plt.ylabel("Treatment Index")
    plt.xticks(range(K))
    plt.yticks(range(K))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_scaf_dendrogram(
    lambda_values,
    all_Z_list,
    round_decimals: int = 2,
    title: str = "",
    save_path: str = "dendrogram.png",
    snap_eps: float | None = None,
    labels=None,
    xlabel: str = "Treatment Index",
    picked_lambda_index: int | None = None,
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

                new_id = next_id
                new_h = lam
                new_mem = cluster_members[a].union(cluster_members[b])

                linkage_list.append([a, b, new_h, len(new_mem)])

                del cluster_members[a]
                del cluster_members[b]
                cluster_members[new_id] = new_mem
                cluster_heights[new_id] = new_h
                next_id += 1

                lst.append(new_id)
                lst.sort(key=lambda x: cluster_heights[x])

    if (not linkage_list) or (not merges_found):
        return None

    # ---- critical fix: complete the tree to exactly K-1 merges ----
    # scipy dendrogram requires a full linkage tree
    if len(linkage_list) < K - 1:
        last_height = float(max(link[2] for link in linkage_list)) if linkage_list else 0.0
        eps = max(1e-8, 1e-6 * max(1.0, last_height))

        remaining = sorted(cluster_members.keys(), key=lambda x: cluster_heights[x])
        while len(remaining) > 1:
            a = remaining.pop(0)
            b = remaining.pop(0)

            new_id = next_id
            new_h = max(cluster_heights[a], cluster_heights[b], last_height) + eps
            new_mem = cluster_members[a].union(cluster_members[b])

            linkage_list.append([a, b, new_h, len(new_mem)])

            del cluster_members[a]
            del cluster_members[b]
            cluster_members[new_id] = new_mem
            cluster_heights[new_id] = new_h
            next_id += 1

            remaining.append(new_id)
            remaining.sort(key=lambda x: cluster_heights[x])

    Z_link = np.array(linkage_list, dtype=float)

    if Z_link.shape[0] != K - 1:
        raise ValueError(
            f"Invalid linkage matrix: got {Z_link.shape[0]} merges, expected {K-1}."
        )

    if labels is None:
        labels = [str(i) for i in range(K)]
    else:
        if len(labels) != K:
            raise ValueError("labels length must equal K.")

    print(f"[Dendrogram] K={K}, len(labels)={len(labels)}, merges={len(linkage_list)}")

    if picked_lambda_index is not None:
        title = (
            f"{title}\n"
            f"(selected lambda idx={picked_lambda_index}, "
            f"lambda={lambda_values[picked_lambda_index]:.4f})"
        )

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"$\lambda$ (fusion strength)")
    sch.dendrogram(Z_link, labels=labels, leaf_rotation=0.0, leaf_font_size=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path