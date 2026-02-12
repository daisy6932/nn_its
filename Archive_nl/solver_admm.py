# Archive_new/solver_admm.py
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

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


def _treatment_indexed_mse(mu_all: torch.Tensor, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    mu_all: [n, M] predicted mean for each treatment
    A:      [n]    int64 treatment labels in {0..M-1}
    y:      [n]    float continuous outcome
    return: scalar MSE between mu_all[i, A_i] and y_i
    """
    device = mu_all.device
    A = A.to(device=device, dtype=torch.long)
    y = y.to(device)

    mu_a = mu_all.gather(1, A.view(-1, 1)).squeeze(1)  # [n]
    return F.mse_loss(mu_a, y)


def run_admm_path(
    model,
    H_train: torch.Tensor,
    A_train: torch.Tensor,
    y_train: torch.Tensor,
    lambda_values: np.ndarray,
    args,
    true_group_labels=None,
    init_Z=None,
    init_U=None,
    log_fn=print,
    # -------- optional --------
    outdir=None,
    save_tag=None,
    save_last_ZU: bool = False,
):
    """
    ADMM over lambda path, optimizing only beta (output_layer.weight).

    Regression setting with categorical treatment A and continuous y:
      mu_all = H_train @ beta.T       -> [n, M]
      loss   = MSE( mu_all[i, A_i], y_i ) + (rho/2)||beta - Z + U||^2

    Returns:
        all_Z: list[np.ndarray]  each element is Z(lam) with shape (M, d_hidden)
        metrics: list[dict]
        Z_final: torch.Tensor
        U_final: torch.Tensor

    If outdir is provided, saves:
        - Z_path_{save_tag}.npy  (shape: [L, M, d_hidden])
        - (optional) Z_last_{save_tag}.npy / U_last_{save_tag}.npy
    """
    device = H_train.device
    rho = float(args.rho)

    # optimizer on beta only
    opt_beta = torch.optim.Adam(model.output_layer.parameters(), lr=float(args.lr_beta))

    # initialize Z, U
    with torch.no_grad():
        beta0 = model.output_layer.weight.detach().clone()

    Z = beta0.clone() if init_Z is None else init_Z.detach().clone().to(device)
    U = torch.zeros_like(Z) if init_U is None else init_U.detach().clone().to(device)

    all_Z = []
    metrics = []

    # safety: ensure types
    if A_train.dtype != torch.long:
        A_train = A_train.long()
    if y_train.dtype not in (torch.float16, torch.float32, torch.float64):
        y_train = y_train.float()

    for li, lam in enumerate(lambda_values):
        lam = float(lam)
        t0 = time.time()

        # ADMM iterations for this lambda
        for _ in range(int(args.admm_epochs)):
            # ---- theta/beta update: approx minimize MSE + (rho/2)||beta - Z + U||^2
            for _ in range(int(args.theta_steps)):
                opt_beta.zero_grad()
                beta = model.output_layer.weight  # [M, d_hidden]

                mu_all = H_train @ beta.T         # [n, M]
                mse = _treatment_indexed_mse(mu_all, A_train, y_train)

                quad = 0.5 * rho * torch.sum((beta - Z + U) ** 2)
                obj = mse + quad
                obj.backward()
                torch.nn.utils.clip_grad_norm_(model.output_layer.parameters(), max_norm=5.0)
                opt_beta.step()

            # ---- Z update
            with torch.no_grad():
                beta_k1 = model.output_layer.weight.detach()
                W = beta_k1 + U

            Z = solve_Z_update(
                W=W,
                lam=lam,
                rho=rho,
                penalty=str(args.penalty),
                gamma_mcp=float(getattr(args, "gamma_mcp", 2.0)),
                a_scad=float(getattr(args, "a_scad", 3.7)),
                z_steps=int(args.z_steps),
                z_lr=float(args.z_lr),
                normalize_fusion=bool(args.normalize_fusion),
                early_tol=float(args.z_early_tol),
                early_patience=int(args.z_early_patience),
                snap_eps=float(args.snap_eps),
            )

            # ---- dual update
            U = U + (beta_k1 - Z)

        # ---- record Z(lam)
        Z_np = Z.detach().cpu().numpy().copy()
        all_Z.append(Z_np)

        # ---- metrics
        c2 = int(np.unique(np.round(Z_np, 2), axis=0).shape[0])
        c3 = int(np.unique(np.round(Z_np, 3), axis=0).shape[0])
        linf = _min_pairwise_linf(Z_np)
        dt = time.time() - t0

        # optional: evaluate ARI of clustering on Z rows (still meaningful)
        ari = None
        if true_group_labels is not None:
            Zr = np.round(Z_np, 3)
            _, cid = np.unique(Zr, axis=0, return_inverse=True)
            ari = float(adjusted_rand_score(true_group_labels, cid))

        msg = (
            f"[ADMM] {li+1:3d}/{len(lambda_values)} "
            f"λ={lam:7.4f} | C2={c2:2d} C3={c3:2d} | "
            f"minLinf={linf:.2e} | T={dt:6.2f}s"
        )
        if ari is not None:
            msg += f" | ARI={ari:.4f}"
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
                "ari_round3": ari,
            }
        )

    # =========================
    # Save Z_path to disk
    # =========================
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

        tag = ""
        if save_tag is not None and len(str(save_tag)) > 0:
            tag = f"_{save_tag}"

        if len(all_Z) > 0:
            Z_path = np.stack(all_Z, axis=0)  # (L, M, d_hidden)
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


