# Archive_new/config.py
import argparse

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # ----- mode/output -----
    parser.add_argument("--mode", type=str, default="posthoc", choices=["posthoc", "alt"])
    parser.add_argument("--out_base", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="exp1")

    # ----- data -----
    parser.add_argument("--n_train", type=int, default=600)
    parser.add_argument("--n_test", type=int, default=10000)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seed_train", type=int, default=0)
    parser.add_argument("--seed_test", type=int, default=42)

    # ----- model -----
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)

    # ----- pretrain / finetune -----
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)

    # finetune settings for alternating
    parser.add_argument("--alt_cycles", type=int, default=3)
    parser.add_argument("--alt_train_epochs", type=int, default=200)
    parser.add_argument("--mu_tether", type=float, default=1.0, help="0.5*mu*||beta - Z||^2 during finetune")

    # ----- lambda path -----
    parser.add_argument("--num_lambda", type=int, default=30)
    parser.add_argument("--max_lambda", type=float, default=12.0)

    # ----- ADMM settings -----
    parser.add_argument("--admm_epochs", type=int, default=60, help="ADMM outer iters per lambda")
    parser.add_argument("--theta_steps", type=int, default=30, help="inner steps for beta/theta update")
    parser.add_argument("--lr_beta", type=float, default=1e-3, help="lr for beta update inside ADMM")

    parser.add_argument("--rho", type=float, default=8.0)

    # ----- Z solver -----
    parser.add_argument("--z_steps", type=int, default=150)
    parser.add_argument("--z_lr", type=float, default=3e-3)
    parser.add_argument("--z_early_tol", type=float, default=1e-4)
    parser.add_argument("--z_early_patience", type=int, default=10)
    parser.add_argument("--snap_eps", type=float, default=1e-3)
    parser.add_argument("--normalize_fusion", action="store_true")

    # ----- penalty choice -----
    parser.add_argument("--penalty", type=str, default="mcp", choices=["l1", "mcp", "scad"])
    parser.add_argument("--gamma_mcp", type=float, default=2.0)
    parser.add_argument("--a_scad", type=float, default=3.7)

    return parser
