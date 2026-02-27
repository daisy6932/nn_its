# Archive_nl/run_scaf_server.py
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
    Z_list = cycle_pack["all_Z"]                  # list of (K,d)
    metrics = cycle_pack["admm_metrics"]

    # 1) save Z_path
    Z_path = np.stack(Z_list, axis=0).astype(np.float32)   # (L,K,d)
    zpath_file = os.path.join(outdir, f"Z_path_cycle{cycle_id}.npy")
    np.save(zpath_file, Z_path)
    log_fn(f"[Save] Z path -> {zpath_file} | shape={Z_path.shape}")

    # 2) save metrics
    metrics_file = os.path.join(outdir, f"path_metrics_cycle{cycle_id}.csv")
    save_path_metrics_csv(metrics, metrics_file)
    log_fn(f"[Save] metrics -> {metrics_file}")

    # 3) save dendrogram（optional: suggest save every cycle）
    dendro_file = os.path.join(outdir, f"dendrogram_cycle{cycle_id}.png")
    dendro = None
    try:
        dendro = plot_scaf_dendrogram(
            lambda_values=lambda_values,
            all_Z_list=Z_list,
            round_decimals=int(getattr(args, "round_decimals", 3)),
            title=f"SCAF Path cycle{cycle_id} ({args.penalty}, rho={args.rho}, gamma={args.gamma_mcp}, a={args.a_scad})",
            save_path=dendro_file
        )
    except Exception as e:
        log_fn(f"[Warn] dendrogram failed in cycle {cycle_id}: {repr(e)}")

    log_fn(f"[Save] dendrogram -> {dendro}")

    #dendro = plot_scaf_dendrogram(
    #    lambda_values=lambda_values,
    #    all_Z_list=Z_list,
        # round_decimals=int(args.round_decimals),
    #    round_decimals=int(getattr(args, "round_decimals", 3)),
    #    title=f"SCAF Path cycle{cycle_id} ({args.penalty}, rho={args.rho}, gamma={args.gamma_mcp}, a={args.a_scad})",
    #    save_path=dendro_file
    #)
    #log_fn(f"[Save] dendrogram -> {dendro}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    add_args(parser)
    # add data_version 
    parser.add_argument(
        "--data_version",
        type=int,
        default=1,
        help="Synthetic data version: 1=linear, 2=weak nonlinear, 3=moderate, 4=strong"
    )
    # add data_version end
    # add 3.0
    parser.add_argument("--round_decimals", type=int, default=3)
    #
    args = parser.parse_args()

    outdir = os.path.join(args.out_base, args.run_name)
    os.makedirs(outdir, exist_ok=True)
    log_fn, log_file = make_logger(os.path.join(outdir, "run.log"))
    log_fn(f"[Run] outdir = {outdir}")
    log_fn(f"[Run] Data Version = {args.data_version}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"--- Using device: {device} ---")

    # data
    if args.n_classes != 10:
        raise ValueError("Synthetic generator assumes n_classes=10.")

    true_groups = [list(range(0, 5)), list(range(5, 10))]
    #X_train, A_train, y_train, _ = generate_synthetic_data(
    #    args.n_train, args.n_features, args.n_classes, true_groups, args.seed_train
    #)
    # add data_version
    X_train, A_train, y_train, _ = generate_synthetic_data(
        args.n_train,
        args.n_features,
        args.n_classes,
        true_groups,
        args.seed_train,
        version=args.data_version
    )

    #X_test, A_test, y_test, _ = generate_synthetic_data(
    #    args.n_test, args.n_features, args.n_classes, true_groups, args.seed_test
    #)
    X_test, A_test, y_test, _ = generate_synthetic_data(
        args.n_test,
        args.n_features,
        args.n_classes,
        true_groups,
        args.seed_test,
        version=args.data_version
    )


    X_train = X_train.to(device); A_train.to(device); y_train = y_train.to(device)
    X_test  = X_test.to(device); A_test = A_test.to(device);  y_test  = y_test.to(device)

    # model
    model = Net(
        n_features=args.n_features,
        n_classes=args.n_classes,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout
    ).to(device)

    # lambda path
    lambda_values = build_lambda_path(args.num_lambda, args.max_lambda)
    np.save(os.path.join(outdir, "lambda_values.npy"), lambda_values)
    log_fn(f"Lambda values (len={len(lambda_values)}): {lambda_values}")

    t0 = time.time()

    if args.mode == "posthoc":
        cycle0_pack, Z_final, U_final = run_posthoc(
            model=model,
            X_train=X_train, A_train=A_train, y_train=y_train,
            X_test=X_test, A_test=A_test, y_test=y_test,
            lambda_values=lambda_values,
            args=args,
            log_fn=log_fn
        )
        save_cycle_artifacts(outdir, 0, lambda_values, cycle0_pack, args, log_fn)

        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump({"mode": "posthoc", "outdir": outdir}, f, indent=2)


    elif args.mode == "alt":

        cycles = run_alternating(

            model=model,

            X_train=X_train, A_train=A_train, y_train=y_train,

            X_test=X_test, A_test=A_test,  y_test=y_test,

            lambda_values=lambda_values,

            args=args,

            log_fn=log_fn

        )

        # cycles 是一个 list，每个元素是 {"cycle":..., "all_Z":..., "admm_metrics":...}

        for pack in cycles:
            cid = int(pack["cycle"])

            save_cycle_artifacts(outdir, cid, lambda_values, pack, args, log_fn)

        with open(os.path.join(outdir, "summary.json"), "w") as f:

            json.dump({"mode": "alt", "cycles": len(cycles), "outdir": outdir}, f, indent=2)


if __name__ == "__main__":
    main()




