# Archive_new/data.py
import torch

def generate_synthetic_data(n_samples, n_features, n_classes, groups, seed):
    """
    Your synthetic: 10 classes split into two groups with different beta patterns.
    """
    if n_classes != 10:
        raise ValueError("This generator currently assumes n_classes=10 (5+5).")

    cpu = torch.device("cpu")
    torch.manual_seed(seed)

    X = -1 + 2 * torch.rand(n_samples, n_features, device=cpu)
    beta_true = torch.zeros(n_classes, n_features, device=cpu)

    group0_beta = torch.tensor([-0.36,  1.8,  1.8] + [0.0]*(n_features-3), device=cpu)
    group1_beta = torch.tensor([ 0.36, -1.8, -1.8] + [0.0]*(n_features-3), device=cpu)

    beta_true[groups[0]] = group0_beta
    beta_true[groups[1]] = group1_beta
    #x,y ycontinous treatment A
    logits = X @ beta_true.T
    y = torch.distributions.Categorical(logits=logits).sample()
    return X, y, beta_true
