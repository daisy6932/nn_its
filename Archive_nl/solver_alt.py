# Archive_new/solver_alt.py
import os
import numpy as np
import torch

from .trainer import train_supervised
from .solver_admm import run_admm_path


def run_posthoc(model, X_train, A_train, y_train, X_test, A_test, y_test,
                lambda_values, args, log_fn=print, outdir=None):
    """
    posthoc (regression w/ treatment):
      Stage 1: train NN to predict y from (X, A) via shared backbone + treatment-specific head
      freeze backbone
      compute H_train = f_theta(X_train)
      Stage 2: ADMM path on beta (output_layer.weight), using (A_train, y_train)
    """
    # -----------------------
    # Stage 1: pretrain NN
    # -----------------------
    log_fn("--- [Stage 1] Pretrain NN (regression) ---")
    train_supervised(
        model,
        X_train, A_train, y_train,
        X_test,  A_test,  y_test,
        epochs=int(args.train_epochs),
        lr=float(args.train_lr),
        wd=float(args.train_wd),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        log_fn=log_fn
    )

    # freeze backbone
    for p in model.fc1.parameters(): p.requires_grad = False
    for p in model.bn1.parameters(): p.requires_grad = False
    for p in model.fc2.parameters(): p.requires_grad = False
    for p in model.bn2.parameters(): p.requires_grad = False

    with torch.no_grad():
        model.eval()
        H_train = model.extract_features(X_train).detach()

    # still meaningful if true groups are known (n_classes=10)
    true_group_labels = np.array([0]*5 + [1]*5) if int(args.n_classes) == 10 else None

    log_fn("--- [Stage 2] ADMM path (posthoc) ---")

    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_Z, metrics, Z_final, U_final = run_admm_path(
        model=model,
        H_train=H_train,
        A_train=A_train,
        y_train=y_train,
        lambda_values=lambda_values,
        args=args,
        true_group_labels=true_group_labels,
        init_Z=None,
        init_U=None,
        log_fn=log_fn,
        outdir=save_dir,
        save_tag="posthoc",
        save_last_ZU=True
    )

    return {
        "cycle": 0,
        "all_Z": all_Z,
        "admm_metrics": metrics,
        "picked_lambda_index": None,
    }, Z_final, U_final


def run_alternating(model, X_train, A_train, y_train, X_test, A_test, y_test,
                    lambda_values, args, log_fn=print, outdir=None):
    """
    alternating (regression w/ treatment):
      Stage 1: pretrain NN
      for each cycle:
        ADMM path -> pick Z*
        finetune NN with tether to Z*
    """
    # -----------------------
    # Stage 1: pretrain NN
    # -----------------------
    log_fn("--- [Stage 1] Pretrain NN (regression) ---")
    train_supervised(
        model,
        X_train, A_train, y_train,
        X_test,  A_test,  y_test,
        epochs=int(args.train_epochs),
        lr=float(args.train_lr),
        wd=float(args.train_wd),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        log_fn=log_fn
    )

    true_group_labels = np.array([0]*5 + [1]*5) if int(args.n_classes) == 10 else None

    Z_init = None
    U_init = None
    all_cycles = []

    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Alt Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        with torch.no_grad():
            model.eval()
            H_train = model.extract_features(X_train).detach()

        all_Z, metrics, Z_final, U_final = run_admm_path(
            model=model,
            H_train=H_train,
            A_train=A_train,
            y_train=y_train,
            lambda_values=lambda_values,
            args=args,
            true_group_labels=true_group_labels,
            init_Z=Z_init,
            init_U=U_init,
            log_fn=log_fn,
            outdir=save_dir,
            save_tag=f"cycle{cycle_id}",
            save_last_ZU=True
        )

        # choose a target Z for finetune
        if true_group_labels is not None:
            best_i = None
            best_ari = -1e9
            for m in metrics:
                if m["ari_round3"] is None:
                    continue
                if m["ari_round3"] > best_ari:
                    best_ari = m["ari_round3"]
                    best_i = int(m["lambda_index"])
            if best_i is None:
                best_i = len(all_Z) - 1
                best_ari = float("nan")
            Z_star_np = all_Z[best_i]
            log_fn(f"[Alt] pick Z* from lambda idx={best_i} (ARI={best_ari:.4f})")
        else:
            best_i = len(all_Z) - 1
            Z_star_np = all_Z[-1]
            log_fn("[Alt] pick Z* from last lambda (no ARI available)")

        Z_star = torch.tensor(Z_star_np, device=X_train.device, dtype=model.output_layer.weight.dtype)

        # unfreeze backbone for finetune
        for p in model.parameters():
            p.requires_grad = True

        log_fn(f"=== [Alt Cycle {cycle_id}] Finetune NN with tether mu={float(args.mu_tether)} ===")
        train_supervised(
            model,
            X_train, A_train, y_train,
            X_test,  A_test,  y_test,
            epochs=int(args.alt_train_epochs),
            lr=float(args.train_lr),
            wd=float(args.train_wd),
            patience=int(args.patience),
            eval_every=int(args.eval_every),
            log_fn=log_fn,
            tether_Z=Z_star,
            mu_tether=float(args.mu_tether)
        )

        # warm start to next cycle
        Z_init = Z_final.detach()
        U_init = U_final.detach()

        all_cycles.append({
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "picked_lambda_index": best_i,
        })

    return all_cycles

