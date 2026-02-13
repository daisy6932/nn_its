# Archive_new/solver_alt.py
import os
import numpy as np
import torch

from .trainer import train_supervised
from .solver_admm import run_admm_path


def run_posthoc(
    model,
    X_train, A_train, y_train,
    X_test,  A_test,  y_test,
    lambda_values,
    args,
    log_fn=print,
    outdir=None
):
    """
    posthoc:
      1) pretrain NN (regression)
      2) freeze backbone
      3) compute H_train
      4) ADMM path on fused parameter (usually gamma)
    """
    # stage 1 pretrain
    log_fn("--- [Stage 1] Pretrain NN ---")
    train_supervised(
        model,
        X_train, A_train, y_train,
        X_test,  A_test,  y_test,
        epochs=int(args.train_epochs),
        lr=float(args.train_lr),
        wd=float(args.train_wd),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        log_fn=log_fn,
        tether_Z=None,
        mu_tether=0.0,
        loss_type=str(getattr(args, "loss_type", "mse")),
        huber_delta=float(getattr(args, "huber_delta", 1.0)),
    )

    # freeze backbone
    for p in model.fc1.parameters(): p.requires_grad = False
    for p in model.bn1.parameters(): p.requires_grad = False
    for p in model.fc2.parameters(): p.requires_grad = False
    for p in model.bn2.parameters(): p.requires_grad = False

    with torch.no_grad():
        model.eval()
        H_train = model.extract_features(X_train).detach()

    # -------- save dir --------
    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    log_fn("--- [Stage 2] ADMM path (posthoc) ---")
    all_Z, metrics, Z_final, U_final = run_admm_path(
        model=model,
        H_train=H_train,
        A_train=A_train,          # NEW
        y_train=y_train,          # NEW (continuous)
        lambda_values=lambda_values,
        args=args,
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


def run_alternating(
    model,
    X_train, A_train, y_train,
    X_test,  A_test,  y_test,
    lambda_values,
    args,
    log_fn=print,
    outdir=None
):
    """
    alternating (博士说的):
      pretrain NN
      for cycle:
        ADMM path -> pick Z*
        finetune NN with tether to Z*
    NOTE:
      - tether 应该绑在 model.gamma.weight (treatment embeddings)
      - ADMM path 也应当优化 gamma / fused parameter
    """
    # stage 1 pretrain
    log_fn("--- [Stage 1] Pretrain NN ---")
    train_supervised(
        model,
        X_train, A_train, y_train,
        X_test,  A_test,  y_test,
        epochs=int(args.train_epochs),
        lr=float(args.train_lr),
        wd=float(args.train_wd),
        patience=int(args.patience),
        eval_every=int(args.eval_every),
        log_fn=log_fn,
        tether_Z=None,
        mu_tether=0.0,
        loss_type=str(getattr(args, "loss_type", "mse")),
        huber_delta=float(getattr(args, "huber_delta", 1.0)),
    )

    Z_init = None
    U_init = None
    all_cycles = []

    # save dir
    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Alt Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        # you can freeze backbone during ADMM if you want (stable)
        if bool(getattr(args, "freeze_backbone_in_admm", True)):
            for p in model.fc1.parameters(): p.requires_grad = False
            for p in model.bn1.parameters(): p.requires_grad = False
            for p in model.fc2.parameters(): p.requires_grad = False
            for p in model.bn2.parameters(): p.requires_grad = False

        with torch.no_grad():
            model.eval()
            H_train = model.extract_features(X_train).detach()

        all_Z, metrics, Z_final, U_final = run_admm_path(
            model=model,
            H_train=H_train,
            A_train=A_train,          # NEW
            y_train=y_train,          # NEW
            lambda_values=lambda_values,
            args=args,
            init_Z=Z_init,
            init_U=U_init,
            log_fn=log_fn,
            outdir=save_dir,
            save_tag=f"cycle{cycle_id}",
            save_last_ZU=True
        )

        # --- pick Z* ---
        # 真实数据：用 validation 曲线挑；synthetic：你也可以用“最接近2组/最稳定”等指标
        best_i = int(np.argmin([m["obj"] for m in metrics])) if ("obj" in metrics[0]) else (len(all_Z) - 1)
        Z_star_np = all_Z[best_i]
        log_fn(f"[Alt] pick Z* from lambda idx={best_i}")

        # IMPORTANT: Z_star 对应 gamma (K x d)
        Z_star = torch.tensor(Z_star_np, device=X_train.device, dtype=model.gamma.weight.dtype)

        # finetune NN (unfreeze backbone)
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
            mu_tether=float(args.mu_tether),
            loss_type=str(getattr(args, "loss_type", "mse")),
            huber_delta=float(getattr(args, "huber_delta", 1.0)),
        )

        # warm start next cycle
        Z_init = Z_final.detach()
        U_init = U_final.detach()

        all_cycles.append({
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "picked_lambda_index": best_i,
        })

    return all_cycles
