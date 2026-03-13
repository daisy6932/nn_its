# Archive_new/solver_admm.py
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from .fusion_solver import solve_Z_update


def _min_pairwise_linf(M: np.ndarray) -> float:
    C = M.shape[0]
    if C < 2:
        return float("inf")
    vals = []
    for i in range(C):
        for j in range(i + 1, C):
            vals.append(np.max(np.abs(M[i] - M[j])))
    return float(np.min(vals)) if vals else float("inf")


def _reg_loss(y_hat: torch.Tensor, y: torch.Tensor, loss_type: str, huber_delta: float) -> torch.Tensor:
    """
    y_hat, y: shape (n,)
    """
    if loss_type == "mse":
        return F.mse_loss(y_hat, y)
    elif loss_type == "huber":
        return F.huber_loss(y_hat, y, delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss_type={loss_type}, use mse|huber")


def run_admm_path(
    model,
    H_train: torch.Tensor,
    A_train: torch.Tensor,        # NEW
    y_train: torch.Tensor,        # continuous y
    lambda_values: np.ndarray,
    args,
    init_Z=None,
    init_U=None,
    log_fn=print,
    outdir: str | None = None,
    save_tag: str | None = None,
    save_last_ZU: bool = False,
):
    """
    ADMM over lambda path, optimizing ONLY treatment parameters gamma (model.gamma.weight).

    Objective (theta-step) for fixed lambda:
        min_gamma  1/n * sum_i loss(y_i, yhat_i(H_i, A_i; gamma)) + (rho/2)||gamma - Z + U||^2

    Z-step:
        Z = argmin_Z  fusion_penalty(Z; lambda) + (rho/2)||Z - (gamma + U)||^2

    Returns:
        all_Z: list[np.ndarray] each element is Z(lam) with shape (K, r)
        metrics: list[dict]
        Z_final, U_final
    """
    device = H_train.device
    rho = float(args.rho)

    # ----------------------------
    # 0) pick which parameter to fuse: gamma
    # ----------------------------
    if not hasattr(model, "gamma"):
        raise AttributeError("Model must have attribute `gamma` (e.g., nn.Embedding(K, r)) for fusion.")

    # gamma_param is a Parameter tensor of shape (K, r)
    gamma_param = model.gamma.weight if hasattr(model.gamma, "weight") else model.gamma
    gamma_param = gamma_param.to(device)

    # optimizer on gamma only
    lr_gamma = float(getattr(args, "lr_gamma", getattr(args, "lr_beta", 1e-3)))
    opt_gamma = torch.optim.Adam([gamma_param], lr=lr_gamma)

    # loss config
    loss_type = str(getattr(args, "loss_type", "mse"))
    huber_delta = float(getattr(args, "huber_delta", 1.0))

    # init Z, U
    with torch.no_grad():
        gamma0 = gamma_param.detach().clone()

    Z = gamma0.clone() if init_Z is None else init_Z.detach().clone().to(device)
    U = torch.zeros_like(Z) if init_U is None else init_U.detach().clone().to(device)

    all_Z: list[np.ndarray] = []
    metrics: list[dict] = []

    # ----------------------------
    # ADMM along lambda path
    # ----------------------------
    for li, lam in enumerate(lambda_values):
        lam = float(lam)
        t0 = time.time()

        for _ in range(int(args.admm_epochs)):
            # ---- theta/gamma update: approx solve regression + quadratic tether to (Z - U)
            for _ in range(int(args.theta_steps)):
                opt_gamma.zero_grad(set_to_none=True)

                # yhat from precomputed H + A
                if hasattr(model, "predict_from_HA"):
                    y_hat = model.predict_from_HA(H_train, A_train)  # (n,)
                else:
                    raise AttributeError(
                        "Model must implement `predict_from_HA(H, A)` for ADMM stage "
                        "(because H_train is precomputed and backbone may be frozen)."
                    )

                reg = _reg_loss(y_hat, y_train, loss_type=loss_type, huber_delta=huber_delta)
                quad = 0.5 * rho * torch.sum((gamma_param - Z + U) ** 2)
                obj = reg + quad

                obj.backward()
                torch.nn.utils.clip_grad_norm_([gamma_param], max_norm=5.0)
                opt_gamma.step()

            # ---- Z update
            with torch.no_grad():
                gamma_k1 = gamma_param.detach()
                W = gamma_k1 + U

            Z = solve_Z_update(
                W=W,
                lam=lam,
                rho=rho,
                penalty=str(args.penalty),
                gamma_mcp=float(getattr(args, "gamma_mcp", 2.0)),
                a_scad=float(getattr(args, "a_scad", 3.7)),
                z_steps=int(args.z_steps),
                z_lr=float(args.z_lr),
                normalize_fusion=bool(getattr(args, "normalize_fusion", False)),
                early_tol=float(getattr(args, "z_early_tol", 1e-4)),
                early_patience=int(getattr(args, "z_early_patience", 6)),
                snap_eps=float(getattr(args, "snap_eps", 1e-3)),
            )

            # ---- dual update
            U = U + (gamma_k1 - Z)

        # ---- record
        Z_np = Z.detach().cpu().numpy().copy()
        all_Z.append(Z_np)

        # ---- metrics
        c2 = int(np.unique(np.round(Z_np, 2), axis=0).shape[0])
        c3 = int(np.unique(np.round(Z_np, 3), axis=0).shape[0])
        linf = _min_pairwise_linf(Z_np)
        dt = time.time() - t0

        # optional: record objective value at end of lambda
        with torch.no_grad():
            if hasattr(model, "predict_from_HA"):
                y_hat_end = model.predict_from_HA(H_train, A_train)
                reg_end = float(_reg_loss(y_hat_end, y_train, loss_type, huber_delta).item())
            else:
                reg_end = float("nan")
        msg = (
            f"[ADMM] {li+1:3d}/{len(lambda_values)} "
            f"λ={lam:7.4f} | C2={c2:2d} C3={c3:2d} | "
            f"minLinf={linf:.2e} | reg={reg_end:.4f} | T={dt:6.2f}s"
        )
        log_fn(msg)

        metrics.append(
            {
                "lambda_index": li,
                "lambda": lam,
                "rho": rho,
                "clusters_2d": c2,
                "clusters_3d": c3,
                "min_linf": linf,
                "time_sec": dt,
                "reg_loss": reg_end,
            }
        )

    # ----------------------------
    # Save Z_path
    # ----------------------------
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        tag = f"_{save_tag}" if (save_tag is not None and len(str(save_tag)) > 0) else ""

        if len(all_Z) > 0:
            Z_path = np.stack(all_Z, axis=0)  # (L, K, r)
            zpath_file = os.path.join(outdir, f"Z_path{tag}.npy")
            np.save(zpath_file, Z_path)
            log_fn(f"[Save] Z_path -> {zpath_file}  (shape={Z_path.shape})")
        else:
            log_fn("[Save] Warning: all_Z empty; did not save Z_path.")

        if save_last_ZU:
            zlast_file = os.path.join(outdir, f"Z_last{tag}.npy")
            ulast_file = os.path.join(outdir, f"U_last{tag}.npy")
            np.save(zlast_file, Z.detach().cpu().numpy())
            np.save(ulast_file, U.detach().cpu().numpy())
            log_fn(f"[Save] Z_last -> {zlast_file}")
            log_fn(f"[Save] U_last -> {ulast_file}")

    return all_Z, metrics, Z.detach(), U.detach()
