# Archive_nl/solver_alt.py
import os
import numpy as np
import torch

from .trainer import train_supervised
from .solver_admm import run_admm_path

def _compute_validation_mse(model, X, A, y):
    """
    Compute treatment-indexed validation MSE:
      1) extract backbone features H
      2) compute mu_all = H @ beta^T
      3) pick mu_all[i, A_i]
      4) compare with y
    """
    device = model.output_layer.weight.device

    with torch.no_grad():
        model.eval()

        X = X.to(device)
        A = A.to(device=device, dtype=torch.long)
        y = y.to(device)

        H = model.extract_features(X)                     # [n, d_hidden]
        beta = model.output_layer.weight                 # [M, d_hidden]
        mu_all = H @ beta.T                              # [n, M]
        mu_a = mu_all.gather(1, A.view(-1, 1)).squeeze(1)

        mse = torch.mean((mu_a - y) ** 2).item()

    return float(mse)


def run_posthoc(model, X_train, A_train, y_train, X_val, A_val, y_val, X_test, A_test, y_test,
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
        X_val, A_val, y_val,
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
    
    A_train = A_train.to(device=H_train.device, dtype=torch.long)
    y_train = y_train.to(H_train.device)


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
    # Week 11
    final_test_mse = _compute_validation_mse(model, X_test, A_test, y_test)
    log_fn(f"[Posthoc Final] test_mse = {final_test_mse:.6f}")


    return {
        "cycle": 0,
        "all_Z": all_Z,
        "admm_metrics": metrics,
        "picked_lambda_index": None,
        "final_test_mse": final_test_mse,
    }, Z_final, U_final


def run_alternating(model, X_train, A_train, y_train, X_val, A_val, y_val, X_test, A_test, y_test,
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
        X_val, A_val, y_val,
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
                      
    # Week 10
    best_val_mse = float("inf") # best validation MSE so far
    best_cycle = None # number of the cycle (best)
    best_state_dict = None
    no_improve_count = 0 # how many times without improving

    alt_early_patience = int(getattr(args, "alt_early_patience", 1)) # how many times allowed without improving
    alt_early_min_delta = float(getattr(args, "alt_early_min_delta", 1e-4)) # Only when the improvement exceeds this threshold is it considered improvement.

    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Alt Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        with torch.no_grad():
            model.eval()
            H_train = model.extract_features(X_train).detach()
        
        A_train = A_train.to(device=H_train.device, dtype=torch.long)
        y_train = y_train.to(H_train.device)


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
                if m["ari_eval"] is None:
                    continue
                if m["ari_eval"] > best_ari:
                    best_ari = m["ari_eval"]
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

        # Z_star = torch.tensor(Z_star_np, device=X_train.device, dtype=model.output_layer.weight.dtype)
        device = model.output_layer.weight.device
        Z_star = torch.tensor(Z_star_np, device=device, dtype=model.output_layer.weight.dtype)


        # unfreeze backbone for finetune
        for p in model.parameters():
            p.requires_grad = True

        log_fn(f"=== [Alt Cycle {cycle_id}] Finetune NN with tether mu={float(args.mu_tether)} ===")
        train_supervised(
            model,
            X_train, A_train, y_train,
            X_val, A_val, y_val,
            epochs=int(args.alt_train_epochs),
            lr=float(args.train_lr),
            wd=float(args.train_wd),
            patience=int(args.patience),
            eval_every=int(args.eval_every),
            log_fn=log_fn,
            tether_Z=Z_star,
            mu_tether=float(args.mu_tether)
        )
        # Week 10
        # validation MSE after this cycle
        val_mse = _compute_validation_mse(model, X_val, A_val, y_val)
        log_fn(f"[Cycle {cycle_id}] validation_mse = {val_mse:.6f}")

        improved = (best_val_mse - val_mse) > alt_early_min_delta

        if improved:
            best_val_mse = val_mse
            best_cycle = cycle_id
            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve_count = 0
            log_fn(
                f"[Cycle {cycle_id}] New best validation MSE: "
                f"{best_val_mse:.6f}"
            )
        else:
            no_improve_count += 1
            log_fn(
                f"[Cycle {cycle_id}] No improvement in validation MSE "
                f"(count={no_improve_count}/{alt_early_patience})"
            )

        # warm start to next cycle
        Z_init = Z_final.detach()
        U_init = U_final.detach()

        all_cycles.append({
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "picked_lambda_index": best_i,
            "val_mse": val_mse, # Week 10
            "best_val_mse_so_far": best_val_mse,
        })

        if no_improve_count >= alt_early_patience:
            log_fn(
                f"[Early Stop] Stop alternating at cycle {cycle_id}. "
                f"Best cycle = {best_cycle}, best validation MSE = {best_val_mse:.6f}"
            )
            break
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log_fn(
            f"[Alt] Restored best model from cycle {best_cycle} "
            f"with validation MSE {best_val_mse:.6f}"
        )
    final_test_mse = _compute_validation_mse(model, X_test, A_test, y_test)
    log_fn(f"[Final] test_mse = {final_test_mse:.6f}")

    if best_cycle is not None:
        log_fn(
            f"[Alt Summary] best_cycle = {best_cycle}, "
            f"best_validation_mse = {best_val_mse:.6f}"
        )
    return all_cycles, {
        "best_cycle": best_cycle,
        "best_val_mse": best_val_mse,
        "final_test_mse": final_test_mse,
    }





