# its_nn/solver_alt.py
import os
import copy
import numpy as np
import torch

from .trainer import train_supervised, eval_mse
from .solver_admm import run_admm_path


def _freeze_backbone(model):
    """
    Freeze feature extractor only.
    """
    for p in model.fc1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.fc2.parameters():
        p.requires_grad = False
    for p in model.bn2.parameters():
        p.requires_grad = False


def _unfreeze_backbone(model):
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.bn1.parameters():
        p.requires_grad = True
    for p in model.fc2.parameters():
        p.requires_grad = True
    for p in model.bn2.parameters():
        p.requires_grad = True


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def _enable_gamma_and_baseline_only(model):
    """
    For stable Phase-1 finetune:
      - freeze everything
      - unfreeze gamma
      - unfreeze m_head if it exists
    """
    _freeze_all(model)

    if hasattr(model, "gamma") and hasattr(model.gamma, "weight"):
        model.gamma.weight.requires_grad = True
    elif hasattr(model, "gamma"):
        for p in model.gamma.parameters():
            p.requires_grad = True

    if getattr(model, "m_head", None) is not None:
        for p in model.m_head.parameters():
            p.requires_grad = True


def _pick_best_lambda(metrics, rule="reg", complexity_weight=0.01, log_fn=print):
    """
    rule options:
      - last
      - middle
      - reg
      - reg_plus_c3
    """
    if len(metrics) == 0:
        raise ValueError("metrics is empty; cannot pick lambda.")

    rule = str(rule).lower()

    if rule == "last":
        best_i = len(metrics) - 1
        log_fn(f"[Alt] lambda_pick=last -> idx={best_i}")
        return best_i

    if rule == "middle":
        best_i = len(metrics) // 2
        log_fn(f"[Alt] lambda_pick=middle -> idx={best_i}")
        return best_i

    if rule == "reg":
        vals = [float(m["reg_loss"]) for m in metrics]
        best_i = int(np.argmin(vals))
        log_fn(f"[Alt] lambda_pick=reg -> idx={best_i}, reg_loss={vals[best_i]:.6f}")
        return best_i

    if rule == "reg_plus_c3":
        scores = []
        for m in metrics:
            reg = float(m["reg_loss"])
            c3 = float(m.get("clusters_3d", 0))
            scores.append(reg + float(complexity_weight) * c3)
        best_i = int(np.argmin(scores))
        log_fn(
            f"[Alt] lambda_pick=reg_plus_c3 -> idx={best_i}, "
            f"score={scores[best_i]:.6f}, "
            f"reg={metrics[best_i]['reg_loss']:.6f}, "
            f"C3={metrics[best_i].get('clusters_3d', 'NA')}"
        )
        return best_i

    raise ValueError(f"Unknown lambda_pick rule={rule}")


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
      4) ADMM path on fused parameter gamma
    """
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

    _freeze_backbone(model)

    with torch.no_grad():
        model.eval()
        H_train = model.extract_features(X_train).detach()

    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    log_fn("--- [Stage 2] ADMM path (posthoc) ---")
    all_Z, metrics, Z_final, U_final = run_admm_path(
        model=model,
        H_train=H_train,
        A_train=A_train,
        y_train=y_train,
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
    More stable alternating scheme:

      Stage 1: pretrain full model

      For each cycle:
        A) freeze backbone and compute H_train
        B) run ADMM path on gamma
        C) pick Z*
        D) finetune in TWO PHASES:
             Phase 1: gamma + m_head only (backbone frozen)
             Phase 2: all params, short refinement
        E) warm start next cycle from Z* (preferred), reset U by default

    Compared with the previous version:
      - less representation drift
      - better respect for the fusion target Z*
      - more stable multi-cycle behavior
    """
    save_dir = outdir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # -----------------------
    # Stage 1: pretrain full model
    # -----------------------
    log_fn("--- [Stage 1] Pretrain NN ---")
    best_val = train_supervised(
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

    best_model_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    Z_init = None
    U_init = None
    all_cycles = []

    # knobs
    lambda_pick_rule = str(getattr(args, "lambda_pick", "reg"))
    complexity_weight = float(getattr(args, "lambda_complexity_weight", 0.01))

    phase1_epochs = int(getattr(args, "alt_phase1_epochs", max(20, int(args.alt_train_epochs) // 2)))
    phase2_epochs = int(getattr(args, "alt_phase2_epochs", max(10, int(args.alt_train_epochs) - phase1_epochs)))

    phase1_lr = float(getattr(args, "alt_phase1_lr", args.train_lr))
    phase2_lr = float(getattr(args, "alt_phase2_lr", args.train_lr * 0.5))

    phase1_wd = float(getattr(args, "alt_phase1_wd", args.train_wd))
    phase2_wd = float(getattr(args, "alt_phase2_wd", args.train_wd))

    disable_cycle_early_stop = bool(getattr(args, "disable_cycle_early_stop", False))
    cycle_patience = int(getattr(args, "cycle_patience", 1))

    warm_start_from_Zstar = bool(getattr(args, "warm_start_from_Zstar", True))
    reset_U_each_cycle = bool(getattr(args, "reset_U_each_cycle", True))

    do_phase2_unfreeze = bool(getattr(args, "alt_phase2_unfreeze", True))

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Alt Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        # -----------------------
        # A) Freeze backbone for ADMM
        # -----------------------
        if bool(getattr(args, "freeze_backbone_in_admm", True)):
            _freeze_backbone(model)
        else:
            _unfreeze_backbone(model)

        with torch.no_grad():
            model.eval()
            H_train = model.extract_features(X_train).detach()

        # -----------------------
        # B) ADMM path on gamma
        # -----------------------
        all_Z, metrics, Z_final, U_final = run_admm_path(
            model=model,
            H_train=H_train,
            A_train=A_train,
            y_train=y_train,
            lambda_values=lambda_values,
            args=args,
            init_Z=Z_init,
            init_U=U_init,
            log_fn=log_fn,
            outdir=save_dir,
            save_tag=f"cycle{cycle_id}",
            save_last_ZU=True
        )

        # -----------------------
        # C) Pick Z*
        # -----------------------
        best_i = _pick_best_lambda(
            metrics=metrics,
            rule=lambda_pick_rule,
            complexity_weight=complexity_weight,
            log_fn=log_fn,
        )
        Z_star_np = all_Z[best_i]
        log_fn(f"[Alt] pick Z* from lambda idx={best_i}, rule={lambda_pick_rule}")

        Z_star = torch.tensor(
            Z_star_np,
            device=X_train.device,
            dtype=model.gamma.weight.dtype
        )

        # -----------------------
        # D) Finetune in TWO PHASES
        # -----------------------

        # Phase 1: gamma + m_head only
        log_fn(
            f"=== [Alt Cycle {cycle_id}] Phase 1 finetune "
            f"(gamma + baseline only), epochs={phase1_epochs}, lr={phase1_lr} ==="
        )
        _enable_gamma_and_baseline_only(model)

        val1 = train_supervised(
            model,
            X_train, A_train, y_train,
            X_test,  A_test,  y_test,
            epochs=phase1_epochs,
            lr=phase1_lr,
            wd=phase1_wd,
            patience=int(args.patience),
            eval_every=int(args.eval_every),
            log_fn=log_fn,
            tether_Z=Z_star,
            mu_tether=float(args.mu_tether),
            loss_type=str(getattr(args, "loss_type", "mse")),
            huber_delta=float(getattr(args, "huber_delta", 1.0)),
        )
        log_fn(f"[Alt] Cycle={cycle_id} Phase1 best_val_mse={val1:.6f}")

        # Phase 2: optionally unfreeze all and refine a bit
        if do_phase2_unfreeze and phase2_epochs > 0:
            log_fn(
                f"=== [Alt Cycle {cycle_id}] Phase 2 finetune "
                f"(all params), epochs={phase2_epochs}, lr={phase2_lr} ==="
            )
            _unfreeze_all(model)

            val2 = train_supervised(
                model,
                X_train, A_train, y_train,
                X_test,  A_test,  y_test,
                epochs=phase2_epochs,
                lr=phase2_lr,
                wd=phase2_wd,
                patience=int(args.patience),
                eval_every=int(args.eval_every),
                log_fn=log_fn,
                tether_Z=Z_star,
                mu_tether=float(args.mu_tether),
                loss_type=str(getattr(args, "loss_type", "mse")),
                huber_delta=float(getattr(args, "huber_delta", 1.0)),
            )
            log_fn(f"[Alt] Cycle={cycle_id} Phase2 best_val_mse={val2:.6f}")

        final_val = eval_mse(model, X_test, A_test, y_test)
        log_fn(f"[Alt] Cycle={cycle_id} final_val_mse={final_val:.6f}")

        # Keep best model state across cycles
        if final_val < best_val - 1e-8:
            best_val = final_val
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            log_fn(f"[Alt] New best model at cycle {cycle_id}: val_mse={best_val:.6f}")
        else:
            no_improve += 1
            log_fn(f"[Alt] No improvement at cycle {cycle_id}; no_improve={no_improve}")

        # -----------------------
        # E) Warm start next cycle
        # -----------------------
        if warm_start_from_Zstar:
            Z_init = Z_star.detach()
        else:
            Z_init = Z_final.detach()

        if reset_U_each_cycle:
            U_init = None
        else:
            U_init = U_final.detach()

        all_cycles.append({
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "picked_lambda_index": best_i,
            "cycle_val_mse": final_val,
        })

        if (not disable_cycle_early_stop) and (no_improve >= cycle_patience):
            log_fn(f"[Alt] Stop alternating early: no improvement for {no_improve} cycle(s).")
            break

    # restore best model found across cycles
    model.load_state_dict(best_model_state)
    log_fn(f"[Alt] Restored best model across cycles with val_mse={best_val:.6f}")

    return all_cycles