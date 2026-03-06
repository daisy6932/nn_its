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
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--seed_train", type=int, default=0)
    parser.add_argument("--seed_test", type=int, default=42)

    # synthetic scenario controls
    parser.add_argument("--scenario", type=str, default="nonlinear_moderate",
                        choices=[
                            "linear_easy",
                            "linear_baseline",
                            "nonlinear_easy",
                            "nonlinear_moderate",
                            "nonlinear_hard",
                            "observational",
                            "misspecified",
                        ])
    parser.add_argument("--y_noise", type=float, default=0.5)
    parser.add_argument("--baseline_scale", type=float, default=1.0)
    parser.add_argument("--effect_scale", type=float, default=1.0)
    parser.add_argument("--near_fused_eps", type=float, default=0.0)

    # ----- model -----
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_baseline", action="store_true")

    # ----- pretrain / finetune -----
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--train_lr", type=float, default=1e-3)
    parser.add_argument("--train_wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)

    parser.add_argument("--alt_cycles", type=int, default=3)
    parser.add_argument("--alt_train_epochs", type=int, default=200)
    parser.add_argument("--mu_tether", type=float, default=1.0)

    # ----- alternating selection / stability -----
    parser.add_argument("--freeze_backbone_in_admm", action="store_true")
    parser.add_argument("--lambda_pick", type=str, default="val",
                        choices=["val", "reg", "last", "middle"])
    parser.add_argument("--reset_U_each_cycle", action="store_true")
    parser.add_argument("--warm_start_from_Zstar", action="store_true")

    # ----- lambda path -----
    parser.add_argument("--num_lambda", type=int, default=30)
    parser.add_argument("--max_lambda", type=float, default=12.0)
    parser.add_argument("--round_decimals", type=int, default=2)

    # ----- ADMM settings -----
    parser.add_argument("--admm_epochs", type=int, default=60)
    parser.add_argument("--theta_steps", type=int, default=30)
    parser.add_argument("--rho", type=float, default=8.0)
    parser.add_argument("--lr_gamma", type=float, default=1e-3)

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

    # ----- loss -----
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)

    parser.add_argument("--cycle_patience", type=int, default=1)
    parser.add_argument("--disable_cycle_early_stop", action="store_true")

    parser.add_argument("--global_seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--alt_phase1_epochs", type=int, default=50)
    parser.add_argument("--alt_phase2_epochs", type=int, default=30)
    parser.add_argument("--alt_phase1_lr", type=float, default=1e-3)
    parser.add_argument("--alt_phase2_lr", type=float, default=5e-4)
    parser.add_argument("--alt_phase1_wd", type=float, default=1e-4)
    parser.add_argument("--alt_phase2_wd", type=float, default=1e-4)
    parser.add_argument("--alt_phase2_unfreeze", action="store_true")

    return parser