# Archive/data.py

import math
import torch
from typing import List, Optional, Sequence, Tuple


def _phi_features(
    X: torch.Tensor,
    version: int = 1,
    alpha_sq: float = 0.1,
    c2: float = 1.0,
    c3: float = 1.0,
    c4: float = 1.0,
) -> torch.Tensor:
    """
    Build feature map Phi(X) depending on version.

    Version 1 (linear):
        Phi = X

    Version 2 (weakly nonlinear):
        Phi = [X, alpha * X^2]

    Version 3 (moderately nonlinear):
        Phi = [X, X^2, sin(pi X), inter(X)]

    Version 4 (strong nonlinear, scaled blocks):
        Phi = [X, c2*X^2, c3*sin(pi X), c4*inter(X)]
    """
    n, d = X.shape
    pi = math.pi

    if version == 1:
        return X

    if version == 2:
        return torch.cat([X, alpha_sq * (X ** 2)], dim=1)

    # Versions 3 & 4 share the same blocks; v4 just scales them.
    x_block = X
    x2_block = (X ** 2)
    sin_block = torch.sin(pi * X)

    blocks = []

    if version == 3:
        blocks.extend([x_block, x2_block, sin_block])
        if d >= 2:
            inter = X[:, :-1] * X[:, 1:]  # (n, d-1)
            blocks.append(inter)
        return torch.cat(blocks, dim=1)

    if version == 4:
        blocks.append(x_block)
        blocks.append(c2 * x2_block)
        blocks.append(c3 * sin_block)
        if d >= 2:
            inter = X[:, :-1] * X[:, 1:]
            blocks.append(c4 * inter)
        return torch.cat(blocks, dim=1)

    raise ValueError(f"Unknown version={version}. Use 1/2/3/4.")


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    groups: List[List[int]],
    seed: int,
    noise_std: float = 0.5,
    treat_probs: Optional[Sequence[float]] = None,
    version: int = 1,
    alpha_sq: float = 0.1,   # used in v2
    c2: float = 1.0,         # used in v4
    c3: float = 1.0,         # used in v4
    c4: float = 1.0,         # used in v4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate regression-style synthetic data:

        X : continuous covariates, shape [n, d], uniform in [-1, 1]
        A : categorical treatment assignment in {0,...,M-1}, shape [n]
        y : continuous outcome, shape [n]
        beta_true : true parameter per treatment, shape [M, p] where p = dim(Phi)

    Data-generating process:
        y_i = < Phi(X_i), beta_{A_i} > + eps_i,   eps_i ~ N(0, noise_std^2)

    Group structure:
        For group 0 treatments: beta_m = g0
        For group 1 treatments: beta_m = g1
    """
    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    n = int(n_samples)
    d = int(n_features)
    M = int(n_classes)

    # 1) X in [-1,1]^d
    X = -1.0 + 2.0 * torch.rand(n, d, device=cpu)

    # 2) treatment assignment A
    if treat_probs is None:
        probs = torch.ones(M, device=cpu) / float(M)
    else:
        probs = torch.tensor(treat_probs, device=cpu, dtype=torch.float32)
        probs = probs / probs.sum()

    A = torch.multinomial(probs, num_samples=n, replacement=True).long()

    # 3) Phi(X)
    Phi = _phi_features(
        X,
        version=version,
        alpha_sq=alpha_sq,
        c2=c2, c3=c3, c4=c4
    )
    p = Phi.shape[1]

    # 4) build group-level true beta (g0, g1), then copy to treatments in that group
    beta_true = torch.zeros(M, p, device=cpu)

    g0 = torch.zeros(p, device=cpu)
    g1 = torch.zeros(p, device=cpu)

    # Put signal in first few linear coordinates (works for all versions)
    k = min(3, p)
    # If d < 3, still fine: it just uses the first k coords.
    g0[:k] = torch.tensor([-0.8, 1.2, 1.0], device=cpu)[:k]
    g1[:k] = torch.tensor([ 0.8,-1.2,-1.0], device=cpu)[:k]

    # Optional: add some signal in the sin-block for nonlinear versions (3/4)
    # This mirrors your earlier logic but is safe across versions.
    if version in (3, 4):
        # For v3/v4, Phi layout is:
        #   [X (d), X^2 (d), sin(pi X) (d), inter (d-1 optional)]
        # sin block starts at 2d, ends at 3d
        sin_start = 2 * d
        if sin_start + 2 <= p:
            g0[sin_start:sin_start + 2] = torch.tensor([0.7, -0.5], device=cpu)
            g1[sin_start:sin_start + 2] = torch.tensor([-0.7, 0.5], device=cpu)

    # Assign beta_true by groups
    # groups is like [[0,1,2,3,4],[5,6,7,8,9]]
    for gi, grp in enumerate(groups):
        for m in grp:
            if m < 0 or m >= M:
                raise ValueError(f"Treatment index {m} out of range for M={M}.")
            beta_true[m] = g0 if gi == 0 else g1

    # 5) y = <Phi(X), beta_{A}> + noise
    y_mean = (Phi * beta_true[A]).sum(dim=1)
    y = y_mean + float(noise_std) * torch.randn(n, device=cpu)

    return X, A, y, beta_true
