# Archive_new/data.py
import torch

def _phi_nonlinear(X: torch.Tensor):
    """
    Simple nonlinear feature map phi(X).
    X: (n, p)
    return: (n, d)
    """
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x3 = X[:, 2:3]

    feats = [
        x1, x2, x3,                      # linear
        x1**2, x2**2,                    # quadratic
        torch.sin(torch.pi * x1),        # sinusoid
        x1 * x2,                         # interaction
    ]
    return torch.cat(feats, dim=1)       # (n, d)

def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    n_treatments: int,
    groups,
    seed: int,
    sigma: float = 1.0,
    treat_random: bool = True,
):
    """
    New DGP:
      X continuous
      A categorical treatment in {0,...,M-1}
      phi(X) nonlinear features
      y continuous outcome

    groups: list of lists, e.g. [[0..4],[5..9]] for M=10
    Returns:
      X: (n,p)
      A: (n,) long
      y: (n,) float
      gamma_true: (M, d) true treatment vectors (fused structure via groups)
    """
    if n_treatments != sum(len(g) for g in groups):
        raise ValueError("n_treatments must equal total size of groups")

    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    # 1) X continuous
    X = -1 + 2 * torch.rand(n_samples, n_features, device=cpu)

    # 2) nonlinear features
    Phi = _phi_nonlinear(X)  # (n, d)
    d = Phi.shape[1]

    # 3) treatment A
    if treat_random:
        A = torch.randint(low=0, high=n_treatments, size=(n_samples,), device=cpu)
    else:
        # observational-ish: A depends on X (optional)
        # simple logits using first 3 covariates
        W = torch.randn(n_treatments, n_features, device=cpu) * 0.3
        logits_A = X @ W.T
        A = torch.distributions.Categorical(logits=logits_A).sample()

    # 4) true treatment vectors gamma_A with group structure (fusion truth)
    gamma_true = torch.zeros(n_treatments, d, device=cpu)

    # example: two groups have different gamma patterns
    g0 = torch.tensor([ 1.0,  0.5, -0.5,  0.8, 0.0, 0.2, 0.4], device=cpu)
    g1 = torch.tensor([-1.0, -0.5,  0.5, -0.8, 0.0,-0.2,-0.4], device=cpu)
    # make sure length matches d
    if g0.numel() != d:
        raise ValueError(f"phi(X) dimension d={d} but gamma templates have len={g0.numel()}")

    gamma_true[groups[0]] = g0
    gamma_true[groups[1]] = g1

    # 5) baseline m(X) + heterogeneous effect <phi(X), gamma_A>
    # baseline: nonlinear but shared
    m = 0.8 * torch.sin(torch.pi * X[:, 0]) + 0.5 * (X[:, 1] ** 2) - 0.3 * X[:, 2]

    # treatment effect term:
    # pick gamma for each sample by A, then inner product with Phi
    gamma_A = gamma_true[A]                     # (n, d)
    tau = torch.sum(Phi * gamma_A, dim=1)       # (n,)

    eps = sigma * torch.randn(n_samples, device=cpu)
    y = m + tau + eps

    return X, A, y, gamma_true
