from data import generate_synthetic_data

X, A, y, beta_true = generate_synthetic_data(
    n_samples=10,
    n_features=10,
    n_classes=10,
    groups=[list(range(0,5)), list(range(5,10))],
    seed=0
)

print("X:", X.shape)
print("A:", A.shape, "unique:", A.unique(sorted=True))
print("y:", y.shape, "mean/std:", y.mean().item(), y.std().item())
print("beta_true:", beta_true.shape)
print("y type:", y.dtype)
