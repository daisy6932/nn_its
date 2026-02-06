# Archive_new/fusion_solver.py
import numpy as np
import torch
from .penalty import fusion_penalty

def snap_rows(Z: torch.Tensor, snap_eps: float) -> torch.Tensor:
    Z_np = Z.detach().cpu().numpy()
    used = set()
    for i in range(Z_np.shape[0]):
        if i in used:
            continue
        group = [i]
        for j in range(i+1, Z_np.shape[0]):
            if np.max(np.abs(Z_np[i] - Z_np[j])) < snap_eps:
                group.append(j)
        if len(group) > 1:
            mean_row = np.mean(Z_np[group], axis=0, keepdims=True)
            Z_np[group] = mean_row
            used.update(group)
    return torch.tensor(Z_np, device=Z.device, dtype=Z.dtype)

def solve_Z_update(
    W: torch.Tensor,
    lam: float,
    rho: float,
    penalty: str,
    gamma_mcp: float,
    a_scad: float,
    z_steps: int,
    z_lr: float,
    normalize_fusion: bool,
    early_tol: float,
    early_patience: int,
    snap_eps: float
) -> torch.Tensor:
    """
    Approx solve:
      Z = argmin_Z  fusion_penalty(Z) + (rho/2)||Z-W||^2
    using Adam on Z (non-convex if MCP/SCAD).
    """
    Z = W.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=z_lr, amsgrad=True)

    last = None
    stall = 0

    for _ in range(z_steps):
        opt.zero_grad()
        pen = fusion_penalty(
            Z, lam,
            penalty=penalty,
            gamma_mcp=gamma_mcp,
            a_scad=a_scad,
            normalize=normalize_fusion
        )
        quad = 0.5 * rho * torch.sum((Z - W)**2)
        obj = pen + quad
        obj.backward()
        opt.step()

        cur = float(obj.item())
        if last is not None and abs(last - cur) <= early_tol * (1.0 + abs(last)):
            stall += 1
            if stall >= early_patience:
                break
        else:
            stall = 0
        last = cur

    with torch.no_grad():
        Z2 = snap_rows(Z, snap_eps)
    return Z2.detach()
