# Archive_nl/plotting.py
import os
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

def plot_scaf_dendrogram(lambda_values, all_Z_list, round_decimals, title, save_path):
    if not all_Z_list:
        return None

    n_classes = all_Z_list[0].shape[0]
    all_clusterings = []
    for Z in all_Z_list:
        Zr = np.round(Z, round_decimals)
        _, cid = np.unique(Zr, axis=0, return_inverse=True)
        all_clusterings.append(cid)

    linkage_list = []
    cluster_members = {i: frozenset([i]) for i in range(n_classes)}
    linkage_ids = {i: i for i in range(n_classes)}
    cluster_heights = {i: 0.0 for i in range(n_classes)}
    next_id = n_classes
    merges_found = False

    # note down the largest lambda
    last_lam = float(lambda_values[-1]) if len(lambda_values) else 1.0

    for i in range(1, len(lambda_values)):
        lam = float(lambda_values[i])
        groups = all_clusterings[i]
        merger_map = {}
        active = list(cluster_members.keys())

        for cid in active:
            item = next(iter(cluster_members[cid]))
            new_gid = groups[item]
            merger_map.setdefault(new_gid, set()).add(cid)

        for _, to_merge in merger_map.items():
            if len(to_merge) <= 1:
                continue

            merges_found = True
            lst = sorted(list(to_merge), key=lambda x: cluster_heights[x])

            while len(lst) > 1:
                # avoid > n-1
                if len(linkage_list) >= n_classes - 1:
                    break

                a = lst.pop(0); b = lst.pop(0)
                la = linkage_ids[a]; lb = linkage_ids[b]

                new = next_id
                new_h = lam
                new_mem = cluster_members[a].union(cluster_members[b])

                linkage_list.append([la, lb, new_h, len(new_mem)])

                del cluster_members[a]; del cluster_members[b]
                cluster_members[new] = new_mem
                linkage_ids[new] = new
                cluster_heights[new] = new_h
                next_id += 1

                lst.append(new)
                lst.sort(key=lambda x: cluster_heights[x])

        
        if len(linkage_list) >= n_classes - 1:
            break

    # If no fusion, do not plot
    if (not linkage_list) or (not merges_found):
        return None

    # make up to n-1 merge
    # SciPy dendrogram  n_classes-1 rows
    # if not fusion, use a relatively larger lamda
    pad_height = last_lam * 1.000001 + 1e-12

    active = sorted(list(cluster_members.keys()), key=lambda x: cluster_heights[x])
    while len(linkage_list) < n_classes - 1 and len(active) > 1:
        a = active.pop(0); b = active.pop(0)
        la = linkage_ids[a]; lb = linkage_ids[b]

        new = next_id
        new_h = max(pad_height, cluster_heights[a], cluster_heights[b])
        new_mem = cluster_members[a].union(cluster_members[b])

        linkage_list.append([la, lb, new_h, len(new_mem)])

        del cluster_members[a]; del cluster_members[b]
        cluster_members[new] = new_mem
        linkage_ids[new] = new
        cluster_heights[new] = new_h
        next_id += 1

        active.append(new)
        active.sort(key=lambda x: cluster_heights[x])

    # last check
    L = np.asarray(linkage_list, dtype=float)
    if L.shape != (n_classes - 1, 4):
        return None

    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Class Index")
    plt.ylabel(r"$\lambda$")
    sch.dendrogram(L, labels=np.arange(n_classes), leaf_rotation=0., leaf_font_size=10)
    plt.savefig(save_path)
    plt.close()
    return save_path
