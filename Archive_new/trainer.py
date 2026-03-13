# Archive_new/trainer.py
import torch
import torch.nn.functional as F


def _eval_mse(model, X, A, y):
    """Validation metric: MSE"""
    model.eval()
    with torch.no_grad():
        y_hat = model(X, A).view(-1)
        y = y.view(-1)
        return float(F.mse_loss(y_hat, y).item())


def train_supervised(
    model,
    X_train, A_train, y_train,
    X_val,   A_val,   y_val,
    epochs: int,
    lr: float,
    wd: float,
    patience: int,
    eval_every: int,
    log_fn=print,
    tether_Z=None,
    mu_tether: float = 0.0,
    loss_type: str = "mse",   # "mse" or "huber"
    huber_delta: float = 1.0
):
    """
    Regression training for y with categorical treatment A.

    Core loss:
      - mse:   L = ||y_hat - y||^2
      - huber: robust alternative

    Optional tether (for alternating scheme):
      L += 0.5 * mu_tether * || gamma - Z ||^2
    where gamma is model.gamma.weight (K x d).
    """
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(wd))

    best = float("inf")
    best_state = None
    bad = 0

    y_train = y_train.view(-1)
    y_val = y_val.view(-1)

    for it in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad()

        y_hat = model(X_train, A_train).view(-1)

        if loss_type == "huber":
            loss = F.huber_loss(y_hat, y_train, delta=float(huber_delta))
        else:
            loss = F.mse_loss(y_hat, y_train)

        # tether on gamma (NOT output_layer anymore)
        if tether_Z is not None and float(mu_tether) > 0.0:
            gamma = model.gamma.weight
            loss = loss + 0.5 * float(mu_tether) * torch.sum((gamma - tether_Z.detach()) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if it % int(eval_every) == 0:
            val_mse = _eval_mse(model, X_val, A_val, y_val)
            log_fn(f"[Train] it={it:5d} | val_mse={val_mse:.6f}")

            if val_mse < best:
                best = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(patience):
                    log_fn(f"[Train] Early stop at it={it}, best_val_mse={best:.6f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best

