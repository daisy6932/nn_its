# its_nn/solver_alt.py
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


def _enable_backbone_only(model, update_baseline=False):
    """
    Freeze everything, then unfreeze backbone only.
    Optionally unfreeze baseline head.
    Gamma stays frozen.
    """
    _freeze_all(model)

    # unfreeze backbone
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.bn1.parameters():
        p.requires_grad = True
    for p in model.fc2.parameters():
        p.requires_grad = True
    for p in model.bn2.parameters():
        p.requires_grad = True

    # optional: unfreeze baseline head
    if update_baseline and getattr(model, "m_head", None) is not None:
        for p in model.m_head.parameters():
            p.requires_grad = True


def _enable_backbone_and_baseline_only(model):
    """
    Freeze everything, then unfreeze backbone + baseline head.
    Gamma stays frozen.
    """
    _enable_backbone_only(model, update_baseline=True)


def _clusters_from_Z_numpy(Z_np, decimals=3):
    """
    Convert a Z matrix (K, d) into cluster ids by rounding rows.
    """
    Zr = np.round(Z_np, decimals)
    _, cid = np.unique(Zr, axis=0, return_inverse=True)
    return cid


def _stability_fro(Z_prev: torch.Tensor, Z_curr: torch.Tensor) -> float:
    """
    Frobenius norm between two selected Z* matrices.
    Smaller means more stable across cycles.
    """
    return float(torch.norm(Z_curr - Z_prev, p="fro").item())


def _stability_labels(Z_prev: torch.Tensor, Z_curr: torch.Tensor, decimals=3) -> float:
    """
    Simple label-based instability:
      fraction of treatments whose rounded-row cluster labels change.
    This is a lightweight proxy, does not require sklearn.
    Smaller means more stable.
    """
    z1 = Z_prev.detach().cpu().numpy()
    z2 = Z_curr.detach().cpu().numpy()

    c1 = _clusters_from_Z_numpy(z1, decimals=decimals)
    c2 = _clusters_from_Z_numpy(z2, decimals=decimals)

    # label ids may be permuted across cycles, so we compare co-membership structure
    K = len(c1)
    disagree = 0
    total = 0
    for i in range(K):
        for j in range(i + 1, K):
            same1 = (c1[i] == c1[j])
            same2 = (c2[i] == c2[j])
            if same1 != same2:
                disagree += 1
            total += 1
    return float(disagree / max(total, 1))


def _compute_cycle_stability(Z_prev, Z_curr, metric="fro", decimals=3):
    metric = str(metric).lower()
    if metric == "fro":
        return _stability_fro(Z_prev, Z_curr)
    if metric == "labels":
        return _stability_labels(Z_prev, Z_curr, decimals=decimals)
    raise ValueError(f"Unknown stability metric={metric}")


def _safe_val_mse(m):
    v = m.get("val_mse", None)
    if v is None:
        raise ValueError("This lambda-pick rule requires val_mse in metrics, but it is missing.")
    return float(v)


def _safe_clusters(m):
    return int(m.get("clusters_3d", 10**9))

def _enable_gamma_only_for_admm(model):
    """
    Prepare model for ADMM step:
      - gamma trainable
      - everything else frozen
    """
    _freeze_all(model)

    if hasattr(model, "gamma") and hasattr(model.gamma, "weight"):
        model.gamma.weight.requires_grad = True
    elif hasattr(model, "gamma"):
        for p in model.gamma.parameters():
            p.requires_grad = True


def _candidate_indices_by_val_tolerance(metrics, tol_abs=None, tol_ratio=None):
    """
    Return indices whose val_mse is within a tolerance of the minimum.
    Use either:
      - tol_abs   : absolute tolerance
      - tol_ratio : relative tolerance, i.e. val <= min_val * (1 + tol_ratio)
    If both provided, a point is kept if it satisfies either threshold.
    """
    vals = np.array([_safe_val_mse(m) for m in metrics], dtype=float)
    min_val = float(np.min(vals))

    keep = []
    for i, v in enumerate(vals):
        ok = False
        if tol_abs is not None:
            ok = ok or (v <= min_val + float(tol_abs))
        if tol_ratio is not None:
            ok = ok or (v <= min_val * (1.0 + float(tol_ratio)))
        if ok:
            keep.append(i)

    if len(keep) == 0:
        # fallback: exact best val
        keep = [int(np.argmin(vals))]

    return keep, min_val


def _pick_best_lambda(
    metrics,
    rule="val_tol_simplest",
    complexity_weight=0.01,
    log_fn=print,
    all_Z=None,
    prev_Z_star=None,
    stability_metric="fro",
    stability_decimals=3,
    val_tol_abs=None,
    val_tol_ratio=0.01,
):
    """
    rule options:
      - last
      - middle
      - reg
      - reg_plus_c3
      - val
      - val_plus_c3
      - val_tol_simplest
      - val_tol_simplest_stable

    Recommended:
      - val_tol_simplest
      - val_tol_simplest_stable   (for alternating runs)

    Notes:
      val_tol_simplest:
        1) find lambdas within tolerance of best val_mse
        2) among them choose smallest clusters_3d
        3) if tie, choose largest lambda index (more regularized / simpler)

      val_tol_simplest_stable:
        same as above, but if prev_Z_star is available, prefer smaller instability
        before using largest lambda index as final tie-break.
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

    if rule == "val":
        vals = [_safe_val_mse(m) for m in metrics]
        best_i = int(np.argmin(vals))
        log_fn(f"[Alt] lambda_pick=val -> idx={best_i}, val_mse={vals[best_i]:.6f}")
        return best_i

    if rule == "val_plus_c3":
        scores = []
        for m in metrics:
            val = _safe_val_mse(m)
            c3 = float(m.get("clusters_3d", 0))
            scores.append(val + float(complexity_weight) * c3)
        best_i = int(np.argmin(scores))
        log_fn(
            f"[Alt] lambda_pick=val_plus_c3 -> idx={best_i}, "
            f"score={scores[best_i]:.6f}, "
            f"val={metrics[best_i]['val_mse']:.6f}, "
            f"C3={metrics[best_i].get('clusters_3d', 'NA')}"
        )
        return best_i

    if rule == "val_tol_simplest":
        candidate_idx, min_val = _candidate_indices_by_val_tolerance(
            metrics,
            tol_abs=val_tol_abs,
            tol_ratio=val_tol_ratio,
        )

        # among acceptable val points, choose simplest (fewest clusters)
        cand_clusters = [_safe_clusters(metrics[i]) for i in candidate_idx]
        min_clusters = int(np.min(cand_clusters))
        best_pool = [i for i in candidate_idx if _safe_clusters(metrics[i]) == min_clusters]

        # final tie-break: choose largest lambda index
        best_i = int(max(best_pool))

        log_fn(
            f"[Alt] lambda_pick=val_tol_simplest -> idx={best_i}, "
            f"min_val={min_val:.6f}, selected_val={_safe_val_mse(metrics[best_i]):.6f}, "
            f"selected_C3={_safe_clusters(metrics[best_i])}, "
            f"candidate_idx={candidate_idx}"
        )
        return best_i

    if rule == "val_tol_simplest_stable":
        candidate_idx, min_val = _candidate_indices_by_val_tolerance(
            metrics,
            tol_abs=val_tol_abs,
            tol_ratio=val_tol_ratio,
        )

        # step 1: simplest among acceptable-val candidates
        cand_clusters = [_safe_clusters(metrics[i]) for i in candidate_idx]
        min_clusters = int(np.min(cand_clusters))
        pool1 = [i for i in candidate_idx if _safe_clusters(metrics[i]) == min_clusters]

        # step 2: if prev_Z_star is available, choose most stable
        if prev_Z_star is not None and all_Z is not None:
            scored = []
            for i in pool1:
                Zi = torch.tensor(
                    all_Z[i],
                    device=prev_Z_star.device,
                    dtype=prev_Z_star.dtype
                )
                stab = _compute_cycle_stability(
                    prev_Z_star,
                    Zi,
                    metric=stability_metric,
                    decimals=stability_decimals
                )
                scored.append((i, stab))

            min_stab = min(s for _, s in scored)
            pool2 = [i for i, s in scored if s <= min_stab + 1e-12]

            # final tie-break: choose largest lambda index
            best_i = int(max(pool2))

            log_fn(
                f"[Alt] lambda_pick=val_tol_simplest_stable -> idx={best_i}, "
                f"min_val={min_val:.6f}, selected_val={_safe_val_mse(metrics[best_i]):.6f}, "
                f"selected_C3={_safe_clusters(metrics[best_i])}, "
                f"selected_stab={min_stab:.6f}, candidate_idx={candidate_idx}, pool1={pool1}"
            )
            return best_i

        # fallback if no prev_Z_star / all_Z
        best_i = int(max(pool1))
        log_fn(
            f"[Alt] lambda_pick=val_tol_simplest_stable(fallback) -> idx={best_i}, "
            f"min_val={min_val:.6f}, selected_val={_safe_val_mse(metrics[best_i]):.6f}, "
            f"selected_C3={_safe_clusters(metrics[best_i])}, "
            f"candidate_idx={candidate_idx}"
        )
        return best_i

    raise ValueError(f"Unknown lambda_pick rule={rule}")


def run_posthoc(
    model,
    X_train, A_train, y_train,
    X_val,   A_val,   y_val,
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
      3) compute H_train / H_val
      4) ADMM path on fused parameter gamma
      5) pick lambda using the same rule as alternating, if desired
    """
    log_fn("--- [Stage 1] Pretrain NN ---")
    train_supervised(
        model,
        X_train, A_train, y_train,
        X_val, A_val, y_val,
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
        H_val = model.extract_features(X_val).detach()

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
        H_val=H_val,
        A_val=A_val,
        y_val=y_val,
        init_Z=None,
        init_U=None,
        log_fn=log_fn,
        outdir=save_dir,
        save_tag="posthoc",
        save_last_ZU=True
    )

    lambda_pick_rule = str(getattr(args, "lambda_pick", "val_tol_simplest"))
    complexity_weight = float(getattr(args, "lambda_complexity_weight", 0.01))
    val_tol_abs = getattr(args, "lambda_val_tol_abs", None)
    val_tol_ratio = float(getattr(args, "lambda_val_tol_ratio", 0.01))

    best_i = _pick_best_lambda(
        metrics=metrics,
        rule=lambda_pick_rule,
        complexity_weight=complexity_weight,
        log_fn=log_fn,
        all_Z=all_Z,
        prev_Z_star=None,
        stability_metric=str(getattr(args, "stability_metric", "fro")),
        stability_decimals=int(getattr(args, "round_decimals", 3)),
        val_tol_abs=val_tol_abs,
        val_tol_ratio=val_tol_ratio,
    )

    return {
        "cycle": 0,
        "all_Z": all_Z,
        "admm_metrics": metrics,
        "picked_lambda_index": best_i,
        "cycle_val_mse": None,
        "cycle_score": None,
        "stability_to_prev": None,
    }, Z_final, U_final


def run_alternating(
    model,
    X_train, A_train, y_train,
    X_val,   A_val,   y_val,
    X_test,  A_test,  y_test,
    lambda_values,
    args,
    log_fn=print,
    outdir=None
):
    """
    More stable alternating scheme for simulation and real-data usage.
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
        X_val, A_val, y_val,
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
        improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
        grad_clip=float(getattr(args, "grad_clip", 5.0)),
    )

    # global best model across cycles
    best_model_state = copy.deepcopy(model.state_dict())
    best_cycle_score = float(best_val)
    best_cycle_id = 0
    no_improve = 0

    Z_init = None
    U_init = None
    prev_Z_star = None
    all_cycles = []

    # knobs
    lambda_pick_rule = str(getattr(args, "lambda_pick", "val_tol_simplest_stable"))
    complexity_weight = float(getattr(args, "lambda_complexity_weight", 0.01))
    val_tol_abs = getattr(args, "lambda_val_tol_abs", None)
    val_tol_ratio = float(getattr(args, "lambda_val_tol_ratio", 0.01))

    phase1_epochs = int(getattr(args, "alt_phase1_epochs", max(20, int(args.alt_train_epochs) // 2)))
    phase2_epochs = int(getattr(args, "alt_phase2_epochs", max(10, int(args.alt_train_epochs) - phase1_epochs)))

    phase1_lr = float(getattr(args, "alt_phase1_lr", args.train_lr))
    phase2_lr = float(getattr(args, "alt_phase2_lr", args.train_lr * 0.5))

    phase1_wd = float(getattr(args, "alt_phase1_wd", args.train_wd))
    phase2_wd = float(getattr(args, "alt_phase2_wd", args.train_wd))

    disable_cycle_early_stop = bool(getattr(args, "disable_cycle_early_stop", False))
    cycle_patience = int(getattr(args, "cycle_patience", 2))
    cycle_improve_tol = float(getattr(args, "cycle_improve_tol", 1e-4))

    warm_start_from_Zstar = bool(getattr(args, "warm_start_from_Zstar", True))
    reset_U_each_cycle = bool(getattr(args, "reset_U_each_cycle", True))

    do_phase2_unfreeze = bool(getattr(args, "alt_phase2_unfreeze", True))

    cycle_metric = str(getattr(args, "cycle_metric", "val_plus_stability")).lower()
    stability_metric = str(getattr(args, "stability_metric", "fro")).lower()
    stability_weight = float(getattr(args, "stability_weight", 0.0))
    stability_tol = float(getattr(args, "stability_tol", 1e-3))
    stability_decimals = int(getattr(args, "round_decimals", 3))

    save_cycle_summary = bool(getattr(args, "save_cycle_summary", False))

    for c in range(int(args.alt_cycles)):
        cycle_id = c + 1
        log_fn(f"\n=== [Alt Cycle {cycle_id}/{int(args.alt_cycles)}] ADMM ===")

        # -----------------------
        # A) Freeze backbone for ADMM
        # -----------------------
        _enable_gamma_only_for_admm(model)
        if bool(getattr(args, "freeze_backbone_in_admm", True)):
            _freeze_backbone(model)
        else:
            _unfreeze_backbone(model)

        with torch.no_grad():
            model.eval()
            H_train = model.extract_features(X_train).detach()
            H_val = model.extract_features(X_val).detach()

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
            H_val=H_val,
            A_val=A_val,
            y_val=y_val,
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
            all_Z=all_Z,
            prev_Z_star=prev_Z_star,
            stability_metric=stability_metric,
            stability_decimals=stability_decimals,
            val_tol_abs=val_tol_abs,
            val_tol_ratio=val_tol_ratio,
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
            X_val, A_val, y_val,
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
            improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
            grad_clip=float(getattr(args, "grad_clip", 5.0)),

        )
        log_fn(f"[Alt] Cycle={cycle_id} Phase1 best_val_mse={val1:.6f}")

        # Phase 2: update representation only (recommended alternating step)
        if do_phase2_unfreeze and phase2_epochs > 0:
            phase2_mode = str(getattr(args, "phase2_mode", "backbone_only")).lower()

            if phase2_mode == "full":
                log_fn(
                    f"=== [Alt Cycle {cycle_id}] Phase 2 finetune "
                    f"(FULL unfreeze; heuristic), epochs={phase2_epochs}, lr={phase2_lr} ==="
                )
                _unfreeze_all(model)

                val2 = train_supervised(
                    model,
                    X_train, A_train, y_train,
                    X_val, A_val, y_val,
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
                    improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                    grad_clip=float(getattr(args, "grad_clip", 5.0)),
                )

            elif phase2_mode == "backbone_only":
                log_fn(
                    f"=== [Alt Cycle {cycle_id}] Phase 2 finetune "
                    f"(backbone only; gamma frozen), epochs={phase2_epochs}, lr={phase2_lr} ==="
                )
                _enable_backbone_only(model, update_baseline=False)

                val2 = train_supervised(
                    model,
                    X_train, A_train, y_train,
                    X_val, A_val, y_val,
                    epochs=phase2_epochs,
                    lr=phase2_lr,
                    wd=phase2_wd,
                    patience=int(args.patience),
                    eval_every=int(args.eval_every),
                    log_fn=log_fn,
                    tether_Z=None,
                    mu_tether=0.0,
                    loss_type=str(getattr(args, "loss_type", "mse")),
                    huber_delta=float(getattr(args, "huber_delta", 1.0)),
                    improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                    grad_clip=float(getattr(args, "grad_clip", 5.0)),
                )

            elif phase2_mode == "backbone_plus_baseline":
                log_fn(
                    f"=== [Alt Cycle {cycle_id}] Phase 2 finetune "
                    f"(backbone + baseline; gamma frozen), epochs={phase2_epochs}, lr={phase2_lr} ==="
                )
                _enable_backbone_and_baseline_only(model)

                val2 = train_supervised(
                    model,
                    X_train, A_train, y_train,
                    X_val, A_val, y_val,
                    epochs=phase2_epochs,
                    lr=phase2_lr,
                    wd=phase2_wd,
                    patience=int(args.patience),
                    eval_every=int(args.eval_every),
                    log_fn=log_fn,
                    tether_Z=None,
                    mu_tether=0.0,
                    loss_type=str(getattr(args, "loss_type", "mse")),
                    huber_delta=float(getattr(args, "huber_delta", 1.0)),
                    improve_tol=float(getattr(args, "train_improve_tol", 1e-6)),
                    grad_clip=float(getattr(args, "grad_clip", 5.0)),
                )

            else:
                raise ValueError(f"Unknown phase2_mode={phase2_mode}")

            log_fn(f"[Alt] Cycle={cycle_id} Phase2 best_val_mse={val2:.6f}")
            final_val = eval_mse(model, X_val, A_val, y_val)
            log_fn(f"[Alt] Cycle={cycle_id} final_val_mse={final_val:.6f}")

        # -----------------------
        # E) Cycle-level score and early stop
        # -----------------------
        stability_to_prev = None
        if prev_Z_star is not None:
            stability_to_prev = _compute_cycle_stability(
                prev_Z_star,
                Z_star,
                metric=stability_metric,
                decimals=stability_decimals
            )
            log_fn(f"[Alt] Cycle={cycle_id} stability_to_prev ({stability_metric}) = {stability_to_prev:.6f}")

        if cycle_metric == "val":
            cycle_score = final_val
        elif cycle_metric == "val_plus_stability":
            cycle_score = final_val
            if stability_to_prev is not None:
                cycle_score = cycle_score + stability_weight * stability_to_prev
        else:
            raise ValueError(f"Unknown cycle_metric={cycle_metric}")

        log_fn(f"[Alt] Cycle={cycle_id} cycle_score={cycle_score:.6f}")

        # Keep best model state across cycles
        improved = (cycle_score < best_cycle_score - cycle_improve_tol)

        if improved:
            best_cycle_score = cycle_score
            best_val = final_val
            best_cycle_id = cycle_id
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            log_fn(
                f"[Alt] New best model at cycle {cycle_id}: "
                f"cycle_score={best_cycle_score:.6f}, val_mse={best_val:.6f}"
            )
        else:
            no_improve += 1
            log_fn(f"[Alt] No cycle-level improvement at cycle {cycle_id}; no_improve={no_improve}")

        # -----------------------
        # F) Warm start next cycle
        # -----------------------
        if warm_start_from_Zstar:
            Z_init = Z_star.detach().clone()
        else:
            Z_init = Z_final.detach().clone()

        if reset_U_each_cycle:
            U_init = None
        else:
            U_init = U_final.detach().clone()

        prev_Z_star = Z_star.detach().clone()

        cycle_record = {
            "cycle": cycle_id,
            "all_Z": all_Z,
            "admm_metrics": metrics,
            "picked_lambda_index": best_i,
            "cycle_val_mse": final_val,
            "cycle_score": cycle_score,
            "stability_to_prev": stability_to_prev,
        }
        all_cycles.append(cycle_record)

        # optional: stop if both no improvement and clustering is already stable
        stable_now = False
        if stability_to_prev is not None:
            stable_now = (stability_to_prev <= stability_tol)

        if (not disable_cycle_early_stop) and (no_improve >= cycle_patience):
            if prev_Z_star is None:
                log_fn(f"[Alt] Stop alternating early: no improvement for {no_improve} cycle(s).")
                break
            else:
                if stable_now or cycle_metric == "val":
                    log_fn(
                        f"[Alt] Stop alternating early: "
                        f"no improvement for {no_improve} cycle(s), "
                        f"stable_now={stable_now}."
                    )
                    break

        # optionally save a compact cycle summary
        if save_cycle_summary and save_dir is not None:
            summary_path = os.path.join(save_dir, f"cycle_summary_{cycle_id}.txt")
            with open(summary_path, "w") as f:
                f.write(f"cycle={cycle_id}\n")
                f.write(f"picked_lambda_index={best_i}\n")
                f.write(f"cycle_val_mse={final_val:.10f}\n")
                f.write(f"cycle_score={cycle_score:.10f}\n")
                f.write(f"stability_to_prev={stability_to_prev}\n")

    # restore best model found across cycles
    model.load_state_dict(best_model_state)
    log_fn(
        f"[Alt] Restored best model across cycles: "
        f"best_cycle={best_cycle_id}, "
        f"best_cycle_score={best_cycle_score:.6f}, "
        f"best_val_mse={best_val:.6f}"
    )

    # best picked lambda index:
    # cycle 0 = pretrain only, so no lambda
    best_picked_lambda_index = None
    if best_cycle_id >= 1:
        best_picked_lambda_index = all_cycles[best_cycle_id - 1].get("picked_lambda_index", None)

    # -----------------------
    # best alternating cycle only (exclude cycle 0 / pretrain)
    # -----------------------
    best_alt_cycle_id = None
    best_alt_cycle_score = None
    best_alt_cycle_val_mse = None
    best_alt_picked_lambda_index = None

    if len(all_cycles) > 0:
        alt_scores = [float(c["cycle_score"]) for c in all_cycles]
        j = int(np.argmin(alt_scores))
        best_alt_cycle = all_cycles[j]

        best_alt_cycle_id = int(best_alt_cycle["cycle"])
        best_alt_cycle_score = float(best_alt_cycle["cycle_score"])
        best_alt_cycle_val_mse = float(best_alt_cycle["cycle_val_mse"])
        best_alt_picked_lambda_index = best_alt_cycle.get("picked_lambda_index", None)

        log_fn(
            f"[Alt] Best alternating-only cycle: "
            f"cycle={best_alt_cycle_id}, "
            f"cycle_score={best_alt_cycle_score:.6f}, "
            f"val_mse={best_alt_cycle_val_mse:.6f}, "
            f"picked_lambda_index={best_alt_picked_lambda_index}"
        )

    return {
        "cycles": all_cycles,
        "best_cycle_id": best_cycle_id,
        "best_cycle_score": best_cycle_score,
        "best_val_mse": best_val,
        "best_picked_lambda_index": best_picked_lambda_index,

        "best_alt_cycle_id": best_alt_cycle_id,
        "best_alt_cycle_score": best_alt_cycle_score,
        "best_alt_cycle_val_mse": best_alt_cycle_val_mse,
        "best_alt_picked_lambda_index": best_alt_picked_lambda_index,
    }