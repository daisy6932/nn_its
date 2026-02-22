# Archive_new/run_scaf_server.py
import os
import json
import time
import numpy as np
import torch

from .config import add_args
from .data import generate_synthetic_data
from .model import Net
from .solver_alt import run_posthoc, run_alternating
from .plotting import save_path_metrics_csv, plot_scaf_dendrogram


def make_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", buffering=1)

    def log_fn(msg):
        s = str(msg)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()

    return log_fn, f


def build_lambda_path(num_lambda, max_lambda):
    small = 1e-4
    num_points = int(num_lambda) - 1
    path = np.geomspace(small, float(max_lambda), num_points)
    return np.concatenate([np.array([0.0]), path])


def save_cycle_artifacts(outdir, cycle_id, lambda_values, cycle_pack, args, log_fn):
    """
    cycle_pack must contain:
      - "all_Z": list of (K,r) np arrays
      - "admm_metrics": list[dict]
    """
    Z_list = cycle_pack["all_Z"]
    metrics = cycle_pack["admm_metrics"]

    # 1) save Z_path
    Z_path = np.stack(Z_list, axis=0).astype(np.float32)  # (L,K,r)
    zpath_file = os.path.join(outdir, f"Z_path_cycle{cycle_id}.npy")
    np.save(zpath_file, Z_path)
    log_fn(f"[Save] Z_path -> {zpath_file} | shape={Z_path.shape}")

    # 2) save metrics
    metrics_file = os.path.join(outdir, f"path_metrics_cycle{cycle_id}.csv")
    save_path_metrics_csv(metrics, metrics_file)
    log_fn(f"[Save] metrics -> {metrics_file}")

    # 3) dendrogram
    dendro_file = os.path.join(outdir, f"dendrogram_cycle{cycle_id}.png")
    dendro = plot_scaf_dendrogram(
        lambda_values=lambda_values,
        all_Z_list=Z_list,
        round_decimals=int(args.round_decimals),
        title=f"SCAF Path cycle{cycle_id} ({args.penalty}, rho={args.rho}, gamma={args.gamma_mcp}, a={args.a_scad})",
        save_path=dendro_file,
        xlabel="Treatment Index"
    )
    log_fn(f"[Save] dendrogram -> {dendro}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    outdir = os.path.join(args.out_base, args.run_name)
    os.makedirs(outdir, exist_ok=True)
    log_fn, log_file = make_logger(os.path.join(outdir, "run.log"))
    log_fn(f"[Run] outdir = {outdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"--- Using device: {device} ---")

    # -------------------------
    # Data (NEW): return X, A, y
    # -------------------------
    # Interpret args.n_actions as number of treatments K (you can rename in config later).
    K = int(args.n_actions)


    # For your original synthetic grouping idea, you still can define groups over treatments:
    if K != 10:
        raise ValueError("This synthetic currently assumes K=10 treatments (5+5).")
    true_groups = [list(range(0, 5)), list(range(5, 10))]

    X_train, A_train, y_train, _ = generate_synthetic_data(
        n_samples=int(args.n_train),
        n_features=int(args.n_features),
        n_treatments=K,  # ✅ 对齐 data.py 的参数名
        groups=true_groups,
        seed=int(args.seed_train),
        sigma=float(args.y_noise),  # ✅ 让 --y_noise 生效
    )
    X_test, A_test, y_test, _ = generate_synthetic_data(
        n_samples=int(args.n_test),
        n_features=int(args.n_features),
        n_treatments=K,  # ✅
        groups=true_groups,
        seed=int(args.seed_test),
        sigma=float(args.y_noise),  # ✅
    )

    X_train = X_train.to(device)
    A_train = A_train.to(device)
    y_train = y_train.to(device)

    X_test = X_test.to(device)
    A_test = A_test.to(device)
    y_test = y_test.to(device)

    # -------------------------
    # Model (NEW): build for treatment regression
    # -------------------------
    # Important: your Net must accept/contain:
    #   - backbone producing H(x)
    #   - treatment embedding gamma(A)
    #   - head producing yhat
    model = Net(
        n_features=int(args.n_features),
        n_treatments=K,  # ✅ 对齐 model.py 的参数名
        hidden1=int(args.hidden1),
        hidden2=int(args.hidden2),
        dropout=float(args.dropout),
    ).to(device)

    # -------------------------
    # Lambda path
    # -------------------------
    lambda_values = build_lambda_path(args.num_lambda, args.max_lambda)
    np.save(os.path.join(outdir, "lambda_values.npy"), lambda_values)
    log_fn(f"Lambda values (len={len(lambda_values)}): {lambda_values}")

    t0 = time.time()

    # -------------------------
    # Run
    # -------------------------
    if args.mode == "posthoc":
        cycle0_pack, Z_final, U_final = run_posthoc(
            model=model,
            X_train=X_train, A_train=A_train, y_train=y_train,
            X_test=X_test,   A_test=A_test,   y_test=y_test,
            lambda_values=lambda_values,
            args=args,
            log_fn=log_fn,
            outdir=outdir,
        )
        save_cycle_artifacts(outdir, 0, lambda_values, cycle0_pack, args, log_fn)

        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump({"mode": "posthoc", "outdir": outdir}, f, indent=2)

    elif args.mode == "alt":
        cycles = run_alternating(
            model=model,
            X_train=X_train, A_train=A_train, y_train=y_train,
            X_test=X_test,   A_test=A_test,   y_test=y_test,
            lambda_values=lambda_values,
            args=args,
            log_fn=log_fn,
            outdir=outdir,
        )

        for pack in cycles:
            cid = int(pack["cycle"])
            save_cycle_artifacts(outdir, cid, lambda_values, pack, args, log_fn)

        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump({"mode": "alt", "cycles": len(cycles), "outdir": outdir}, f, indent=2)

    else:
        raise ValueError(f"Unknown mode={args.mode}")

    log_fn(f"[Done] total_time_sec={time.time()-t0:.2f}")
    log_file.close()


if __name__ == "__main__":
    main()

