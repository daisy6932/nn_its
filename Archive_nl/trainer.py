# Archive_new/trainer.py
import torch
import torch.nn.functional as F

def _eval_mse(model, X, A, y):
    """
    Evaluate MSE on (X, A, y):
      model(X) -> [n, M] predicted means for each treatment
      pick the A-th column for each sample -> [n]
      compute MSE with continuous y -> scalar
    """
    model.eval()
    with torch.no_grad():
        mu = model(X)  # [n, M]
        mu_a = mu.gather(1, A.view(-1, 1)).squeeze(1)  # [n]
        return float(F.mse_loss(mu_a, y).item())

def train_supervised(
    model,
    X_train, A_train, y_train,
    X_val, A_val, y_val,
    epochs: int,
    lr: float,
    wd: float,
    patience: int,
    eval_every: int,
    log_fn=print,
    tether_Z=None,
    mu_tether: float = 0.0
):
    """
    Supervised training for continuous outcome with categorical treatment A:
      model(X) returns [n, M] = predicted mean outcome under each treatment
      loss = MSE( model(X)[i, A_i], y_i ) + 0.5*mu*||beta - Z||^2 (optional tether)
    where beta is output_layer.weight (shape [M, hidden2]).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = float("inf")
    best_state = None
    bad = 0

    for it in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        mu = model(X_train)  # [n, M]
        mu_a = mu.gather(1, A_train.view(-1, 1)).squeeze(1)  # [n]
        loss = F.mse_loss(mu_a, y_train)

        if tether_Z is not None and mu_tether > 0:
            beta = model.output_layer.weight
            loss = loss + 0.5 * mu_tether * torch.sum((beta - tether_Z.detach()) ** 2)

        loss.backward()
        opt.step()

        if it % eval_every == 0:
            val_mse = _eval_mse(model, X_val, A_val, y_val)
            log_fn(f"[Train] it={it:5d} | val_mse={val_mse:.6f}")

            if val_mse < best:
                best = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log_fn(f"[Train] Early stop at it={it}, best_val_mse={best:.6f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best

