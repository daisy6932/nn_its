import math
import torch
from typing import List, Tuple, Optional


def _phi_linear(X: torch.Tensor) -> torch.Tensor:
    return X


def _phi_nonlinear_basic(X: torch.Tensor) -> torch.Tensor:
    """
    Moderate nonlinear feature map.
    X: (n, p), assumes p >= 5 for richest behavior, but works for p >= 3.
    """
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]

    feats = [
        x1, x2, x3,
        x1 ** 2, x2 ** 2,
        torch.sin(math.pi * x1),
        x1 * x2,
    ]
    return torch.cat(feats, dim=1)


def _phi_nonlinear_richer(X: torch.Tensor) -> torch.Tensor:
    """
    Richer nonlinear map; still low-dimensional enough for interpretation.
    """
    n, p = X.shape
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]
    x4 = X[:, 3:4] if p >= 4 else X[:, 0:1]
    x5 = X[:, 4:5] if p >= 5 else X[:, 1:2]

    feats = [
        x1, x2, x3,
        x1 ** 2, x2 ** 2, x3 ** 2,
        torch.sin(math.pi * x1),
        torch.cos(math.pi * x2),
        x1 * x2,
        x2 * x3,
        x4 * x5,
    ]
    return torch.cat(feats, dim=1)


def _baseline_function(X: torch.Tensor, scenario: str, baseline_scale: float) -> torch.Tensor:
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    if scenario in ["linear_easy", "linear_baseline"]:
        m = 0.8 * x1 - 0.5 * x2 + 0.3 * x3
    elif scenario in ["nonlinear_easy", "nonlinear_moderate", "observational", "misspecified"]:
        m = 0.6 * torch.cos(math.pi * x1) + 0.4 * (x2 ** 2) - 0.3 * x3
    elif scenario == "nonlinear_hard":
        m = (
            0.7 * torch.cos(math.pi * x1)
            + 0.5 * torch.sin(math.pi * x2)
            + 0.3 * (x3 ** 2)
            - 0.2 * x1 * x2
        )
    else:
        raise ValueError(f"Unknown scenario={scenario}")

    return baseline_scale * m


def _make_true_gamma(
    n_treatments: int,
    d_phi: int,
    groups: List[List[int]],
    scenario: str,
    effect_scale: float,
    near_fused_eps: float = 0.0,
) -> torch.Tensor:
    """
    Construct true treatment-specific vectors gamma_true with grouped structure.
    """
    gamma_true = torch.zeros(n_treatments, d_phi)

    # two prototype group vectors
    g0 = torch.zeros(d_phi)
    g1 = torch.zeros(d_phi)

    if scenario in ["linear_easy", "linear_baseline"]:
        k = min(3, d_phi)
        g0[:k] = torch.tensor([1.0, 0.6, -0.6])[:k]
        g1[:k] = torch.tensor([-1.0, -0.6, 0.6])[:k]

    elif scenario in ["nonlinear_easy", "nonlinear_moderate", "observational", "misspecified"]:
        k = min(7, d_phi)
        g0[:k] = torch.tensor([1.0, 0.5, -0.5, 0.8, 0.0, 0.2, 0.4])[:k]
        g1[:k] = torch.tensor([-1.0, -0.5, 0.5, -0.8, 0.0, -0.2, -0.4])[:k]

    elif scenario == "nonlinear_hard":
        k = min(11, d_phi)
        g0[:k] = torch.tensor([1.0, 0.6, -0.4, 0.8, 0.2, 0.4, 0.3, -0.2, 0.5, 0.2, -0.3])[:k]
        g1[:k] = torch.tensor([-0.9, -0.5, 0.5, -0.7, -0.1, -0.3, -0.2, 0.2, -0.4, -0.2, 0.3])[:k]
    else:
        raise ValueError(f"Unknown scenario={scenario}")

    g0 = effect_scale * g0
    g1 = effect_scale * g1

    for gi, grp in enumerate(groups):
        proto = g0 if gi == 0 else g1
        for t in grp:
            gamma_true[t] = proto.clone()

            # optional: allow approximate fusion instead of exact equality
            if near_fused_eps > 0:
                gamma_true[t] += near_fused_eps * torch.randn(d_phi)

    return gamma_true


def _sample_treatment(
    X: torch.Tensor,
    n_treatments: int,
    treat_random: bool,
    seed: int,
) -> torch.Tensor:
    """
    Sample treatment A.
    If treat_random=True: randomized trial style.
    Else: observational assignment depending on X.
    """
    torch.manual_seed(seed + 12345)
    n, p = X.shape

    if treat_random:
        return torch.randint(low=0, high=n_treatments, size=(n,), device=X.device)

    W = torch.randn(n_treatments, p, device=X.device) * 0.35
    logits = X @ W.T
    return torch.distributions.Categorical(logits=logits).sample()


def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_treatments: int,
    groups: List[List[int]],
    seed: int,
    scenario: str = "nonlinear_moderate",
    sigma: float = 0.5,
    treat_random: bool = True,
    baseline_scale: float = 1.0,
    effect_scale: float = 1.0,
    near_fused_eps: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified synthetic DGP for both simulation stress tests and method development.

    Returns
    -------
    X : (n, p)
    A : (n,)
    y : (n,)
    gamma_true : (K, d_phi)

    Scenarios
    ---------
    linear_easy
    linear_baseline
    nonlinear_easy
    nonlinear_moderate
    nonlinear_hard
    observational
    misspecified
    """
    if n_treatments != sum(len(g) for g in groups):
        raise ValueError("n_treatments must equal total size of groups")

    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    X = -1.0 + 2.0 * torch.rand(n_samples, n_features, device=cpu)

    # choose phi
    if scenario in ["linear_easy", "linear_baseline"]:
        Phi = _phi_linear(X[:, :min(3, n_features)])
    elif scenario in ["nonlinear_easy", "nonlinear_moderate", "observational"]:
        Phi = _phi_nonlinear_basic(X)
    elif scenario in ["nonlinear_hard", "misspecified"]:
        Phi = _phi_nonlinear_richer(X)
    else:
        raise ValueError(f"Unknown scenario={scenario}")

    d_phi = Phi.shape[1]

    # observational scenario should usually use treatment depending on X
    if scenario == "observational":
        treat_random = False

    A = _sample_treatment(X, n_treatments=n_treatments, treat_random=treat_random, seed=seed)

    gamma_true = _make_true_gamma(
        n_treatments=n_treatments,
        d_phi=d_phi,
        groups=groups,
        scenario=scenario,
        effect_scale=effect_scale,
        near_fused_eps=near_fused_eps,
    ).to(cpu)

    m = _baseline_function(X, scenario=scenario, baseline_scale=baseline_scale)
    gamma_A = gamma_true[A]
    tau = torch.sum(Phi * gamma_A, dim=1)

    eps = sigma * torch.randn(n_samples, device=cpu)
    y = m + tau + eps

    return X, A.long(), y.float(), gamma_true