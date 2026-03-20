# its_nn/trainer.py
import torch
import torch.nn.functional as F


def eval_mse(model, X, A, y):
    """
    Evaluate prediction MSE only.
    """
    model.eval()
    with torch.no_grad():
        y_hat = model(X, A).view(-1)
        y = y.view(-1)
        return float(F.mse_loss(y_hat, y).item())


def _prediction_loss(y_hat, y, loss_type="mse", huber_delta=1.0):
    loss_type = str(loss_type).lower()
    if loss_type == "huber":
        return F.huber_loss(y_hat, y, delta=float(huber_delta))
    elif loss_type == "mse":
        return F.mse_loss(y_hat, y)
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")


def _tether_penalty(model, tether_Z):
    """
    Return raw tether penalty ||gamma - tether_Z||^2 / 2 form without mu.
    """
    gamma = model.gamma.weight
    return 0.5 * torch.sum((gamma - tether_Z.detach()) ** 2)


def _gamma_distance(model, tether_Z):
    """
    Frobenius distance ||gamma - tether_Z||_F.
    """
    gamma = model.gamma.weight
    return float(torch.norm(gamma - tether_Z.detach(), p="fro").item())


def _trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


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
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    improve_tol: float = 1e-6,
    grad_clip: float = 5.0,
):
    """
    Supervised training for outcome regression.

    Optimizes:
        pred_loss + mu_tether * 0.5 * ||gamma - tether_Z||^2

    Returns:
        best_val_mse
    """
    params = _trainable_parameters(model)
    if len(params) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad flags before training.")

    opt = torch.optim.Adam(params, lr=float(lr), weight_decay=float(wd))

    best = float("inf")
    best_state = None
    bad = 0

    y_train = y_train.view(-1)
    y_val = y_val.view(-1)

    for it in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad()

        y_hat = model(X_train, A_train).view(-1)

        pred_loss = _prediction_loss(
            y_hat, y_train,
            loss_type=loss_type,
            huber_delta=huber_delta
        )

        tether_pen = torch.tensor(0.0, device=y_hat.device)
        gamma_dist = None

        if tether_Z is not None and float(mu_tether) > 0.0:
            tether_pen = _tether_penalty(model, tether_Z)
            total_loss = pred_loss + float(mu_tether) * tether_pen
            gamma_dist = _gamma_distance(model, tether_Z)
        else:
            total_loss = pred_loss

        total_loss.backward()

        if grad_clip is not None and float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))

        opt.step()

        if it % int(eval_every) == 0 or it == 1 or it == int(epochs):
            val_mse = eval_mse(model, X_val, A_val, y_val)

            msg = (
                f"[Train] it={it:5d} | "
                f"train_pred_loss={float(pred_loss.item()):.6f} | "
                f"train_total_loss={float(total_loss.item()):.6f} | "
                f"val_mse={val_mse:.6f}"
            )

            if tether_Z is not None and float(mu_tether) > 0.0:
                msg += (
                    f" | tether_pen={float(tether_pen.item()):.6f}"
                    f" | gamma_dist={float(gamma_dist):.6f}"
                )

            log_fn(msg)

            if val_mse < best - float(improve_tol):
                best = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
                log_fn(f"[Train] New best at it={it:5d} | best_val_mse={best:.6f}")
            else:
                bad += 1
                if bad >= int(patience):
                    log_fn(
                        f"[Train] Early stop at it={it:5d} | "
                        f"best_val_mse={best:.6f} | "
                        f"no_improve_count={bad}"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log_fn("[Train] Warning: best_state is None, model kept at final iterate.")

    return best