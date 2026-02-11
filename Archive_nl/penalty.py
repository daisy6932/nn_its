# Archive_new/penalty.py
import itertools
import torch

def _normalize(pen, C, d, normalize: bool):
    if normalize and C > 1 and d > 0:
        pen = pen / (C*(C-1)/2.0) / d
    return pen

def fusion_penalty_l1(beta, lam, normalize=False):
    C, d = beta.shape
    pen = beta.new_tensor(0.0)
    for q, k in itertools.combinations(range(C), 2):
        pen = pen + torch.sum(torch.abs(beta[q] - beta[k])) * lam
    return _normalize(pen, C, d, normalize)

def fusion_penalty_elementwise_mcp(beta, lam, gamma_mcp, normalize=False):
    """
    MCP on elementwise absolute differences.
    """
    C, d = beta.shape
    pen = beta.new_tensor(0.0)
    for q, k in itertools.combinations(range(C), 2):
        a = torch.abs(beta[q] - beta[k])
        val = lam * a - (a * a) / (2.0 * gamma_mcp)
        cap = 0.5 * gamma_mcp * lam * lam
        pen = pen + torch.sum(torch.where(a <= gamma_mcp * lam, val, cap))
    return _normalize(pen, C, d, normalize)

def fusion_penalty_elementwise_scad(beta, lam, a_scad=3.7, normalize=False):
    """
    SCAD on elementwise absolute differences.
    Piecewise:
      p'(t)=lam for t<=lam
      p'(t)=(a*lam - t)/(a-1) for lam<t<=a*lam
      p'(t)=0 for t>a*lam
    Integrated closed form:
      p(t)= lam*t                              , t<=lam
           ( -t^2 + 2*a*lam*t - lam^2 )/(2*(a-1)) , lam<t<=a*lam
           (lam^2*(a+1))/2                     , t>a*lam
    """
    C, d = beta.shape
    pen = beta.new_tensor(0.0)
    a = float(a_scad)

    for q, k in itertools.combinations(range(C), 2):
        t = torch.abs(beta[q] - beta[k])

        part1 = lam * t
        part2 = (-t*t + 2*a*lam*t - lam*lam) / (2.0*(a-1.0))
        part3 = (lam*lam*(a+1.0))/2.0

        pen = pen + torch.sum(torch.where(
            t <= lam,
            part1,
            torch.where(t <= a*lam, part2, part3)
        ))

    return _normalize(pen, C, d, normalize)

def fusion_penalty(beta, lam, penalty="mcp", gamma_mcp=2.0, a_scad=3.7, normalize=False):
    if penalty == "l1":
        return fusion_penalty_l1(beta, lam, normalize)
    if penalty == "mcp":
        return fusion_penalty_elementwise_mcp(beta, lam, gamma_mcp, normalize)
    if penalty == "scad":
        return fusion_penalty_elementwise_scad(beta, lam, a_scad, normalize)
    raise ValueError(f"Unknown penalty={penalty}")
