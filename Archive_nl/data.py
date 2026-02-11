# src/data.py
import torch
import math

def nonlinear_features(X: torch.Tensor) -> torch.Tensor:
    """
    input: X [n, d]
    ouput: Phi [n, p]  (nonlinear)
     non-linear basis:
      - x
      - x^2
      - sin(pi x)
      - interactions x_i * x_{i+1}
    """
    n, d = X.shape
    pi = math.pi

    phi_list = []
    phi_list.append(X)                    # [n, d]
    phi_list.append(X ** 2)               # [n, d]
    phi_list.append(torch.sin(pi * X))    # [n, d]

    if d >= 2:
        inter = X[:, :-1] * X[:, 1:]      # [n, d-1]
        phi_list.append(inter)

    Phi = torch.cat(phi_list, dim=1)      # [n, p]
    return Phi


def generate_synthetic_data(n_samples, n_features, n_classes, groups, seed,
                            noise_std=0.5, treat_probs=None):
    """
    generate regression data:
      X: continuous features, shape [n, d]
      A: categorical treatment (0..M-1), shape [n]
      y: continuous outcome, shape [n]
      beta_true: true parameter for every treatment , shape [M, p]
    """
    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    n = n_samples
    d = n_features
    M = n_classes  # treatment count

    # 1) X continuous
    X = -1 + 2 * torch.rand(n, d, device=cpu)

    # 2) A categorical (treatment assignment)
    if treat_probs is None:
        treat_probs = torch.ones(M, device=cpu) / M
    else:
        treat_probs = torch.tensor(treat_probs, device=cpu, dtype=torch.float32)
        treat_probs = treat_probs / treat_probs.sum()

    A = torch.multinomial(treat_probs, num_samples=n, replacement=True).long()

    # 3) nonlinear map Phi(X)
    Phi = nonlinear_features(X)           # [n, p]
    p = Phi.shape[1]

    # 4) build group-level true beta, then copy to treatments in that group
    beta_true = torch.zeros(M, p, device=cpu)

    g0 = torch.zeros(p, device=cpu)
    g1 = torch.zeros(p, device=cpu)

    g0[:3] = torch.tensor([-0.8, 1.2, 1.0], device=cpu)
    g1[:3] = torch.tensor([ 0.8,-1.2,-1.0], device=cpu)

    sin_start = 2*d
    sin_end = 3*d
    if sin_end <= p:
        g0[sin_start:sin_start+2] = torch.tensor([0.7, -0.5], device=cpu)
        g1[sin_start:sin_start+2] = torch.tensor([-0.7, 0.5], device=cpu)

    # assign to each treatment by group
    for gi, grp in enumerate(groups):
        for m in grp:
            beta_true[m] = g0 if gi == 0 else g1

    # 5) y continuous: y = Phi(X) dot beta_{A} + noise
    y_mean = (Phi * beta_true[A]).sum(dim=1)      # [n]
    y = y_mean + noise_std * torch.randn(n, device=cpu)

    return X, A, y, beta_true
