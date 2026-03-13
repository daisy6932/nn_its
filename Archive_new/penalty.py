# Archive_new/penalty.py
import torch


def _normalize(pen: torch.Tensor, C: int, d: int, normalize: bool) -> torch.Tensor:
    if normalize and C > 1 and d > 0:
        pen = pen / (C * (C - 1) / 2.0) / d
    return pen


def _pairwise_diffs(beta: torch.Tensor) -> torch.Tensor:
    """
    Return all pairwise differences beta[q]-beta[k] for q<k as a tensor of shape (P, d),
    where P = C*(C-1)/2.
    """
    C, d = beta.shape
    idx_q, idx_k = torch.triu_indices(C, C, offset=1, device=beta.device)
    diffs = beta[idx_q] - beta[idx_k]   # (P, d)
    return diffs


def fusion_penalty_l1(beta: torch.Tensor, lam: float, normalize: bool = False) -> torch.Tensor:
    C, d = beta.shape
    diffs = _pairwise_diffs(beta)
    pen = lam * torch.abs(diffs).sum()
    return _normalize(pen, C, d, normalize)


def fusion_penalty_elementwise_mcp(
    beta: torch.Tensor,
    lam: float,
    gamma_mcp: float,
    normalize: bool = False
) -> torch.Tensor:
    """
    Elementwise MCP on |beta[q]-beta[k]|.

    p(t) = lam*t - t^2/(2*gamma)   if t <= gamma*lam
           0.5*gamma*lam^2        if t >  gamma*lam
    """
    C, d = beta.shape
    diffs = _pairwise_diffs(beta)
    a = torch.abs(diffs)

    lam_t = lam * a
    quad = (a * a) / (2.0 * gamma_mcp)
    val = lam_t - quad
    cap = 0.5 * gamma_mcp * lam * lam

    pen = torch.where(a <= gamma_mcp * lam, val, cap).sum()
    return _normalize(pen, C, d, normalize)


def fusion_penalty_elementwise_scad(
    beta: torch.Tensor,
    lam: float,
    a_scad: float = 3.7,
    normalize: bool = False
) -> torch.Tensor:
    """
    Elementwise SCAD on |beta[q]-beta[k]|.

    Closed-form:
      p(t)= lam*t                                        , t<=lam
           ( -t^2 + 2*a*lam*t - lam^2 )/(2*(a-1))        , lam<t<=a*lam
           (lam^2*(a+1))/2                               , t>a*lam
    """
    C, d = beta.shape
    diffs = _pairwise_diffs(beta)
    t = torch.abs(diffs)
    a = float(a_scad)

    part1 = lam * t
    part2 = (-t * t + 2.0 * a * lam * t - lam * lam) / (2.0 * (a - 1.0))
    part3 = (lam * lam * (a + 1.0)) / 2.0

    pen_elem = torch.where(t <= lam, part1, torch.where(t <= a * lam, part2, part3))

    # 数值稳定：极少数情况下 part2 会因为浮点误差出现 아주小负数
    pen_elem = torch.clamp(pen_elem, min=0.0)

    pen = pen_elem.sum()
    return _normalize(pen, C, d, normalize)


def fusion_penalty(
    beta: torch.Tensor,
    lam: float,
    penalty: str = "mcp",
    gamma_mcp: float = 2.0,
    a_scad: float = 3.7,
    normalize: bool = False
) -> torch.Tensor:
    penalty = str(penalty).lower()
    if penalty == "l1":
        return fusion_penalty_l1(beta, lam, normalize)
    if penalty == "mcp":
        return fusion_penalty_elementwise_mcp(beta, lam, gamma_mcp, normalize)
    if penalty == "scad":
        return fusion_penalty_elementwise_scad(beta, lam, a_scad, normalize)
    raise ValueError(f"Unknown penalty={penalty}")
