# Archive_new/trainer.py
import torch
import torch.nn.functional as F

def _eval_nll(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        return float(F.cross_entropy(logits, y).item())

def train_supervised(
    model,
    X_train, y_train,
    X_val, y_val,
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
    Standard NN training with optional tether:
      loss = CE + 0.5*mu*||beta - Z||^2
    where beta is output_layer.weight.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best = float("inf")
    best_state = None
    bad = 0

    for it in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = F.cross_entropy(logits, y_train)

        if tether_Z is not None and mu_tether > 0:
            beta = model.output_layer.weight
            loss = loss + 0.5 * mu_tether * torch.sum((beta - tether_Z.detach())**2)

        loss.backward()
        opt.step()

        if it % eval_every == 0:
            val_nll = _eval_nll(model, X_val, y_val)
            log_fn(f"[Train] it={it:5d} | val_nll={val_nll:.4f}")

            if val_nll < best:
                best = val_nll
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    log_fn(f"[Train] Early stop at it={it}, best_val_nll={best:.4f}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best
