# Archive_new/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Regression with categorical treatment A:
        h = H(x) in R^d
        gamma[a] in R^d  (to be fused across treatments)
        y_hat = m(x) + <h, gamma[a]>
    """
    def __init__(self, n_features, n_treatments, hidden1=32, hidden2=16, dropout=0.3, use_baseline=True):
        super().__init__()
        self.n_treatments = int(n_treatments)
        self.hidden2 = int(hidden2)
        self.use_baseline = bool(use_baseline)

        # backbone
        self.fc1 = nn.Linear(n_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout = nn.Dropout(p=dropout)

        # treatment-specific parameter to be fused: gamma[a] in R^hidden2
        # Using Embedding is convenient: gamma.weight has shape (K, d)
        self.gamma = nn.Embedding(self.n_treatments, hidden2)

        # optional baseline m(x)
        if self.use_baseline:
            self.m_head = nn.Linear(hidden2, 1, bias=True)
        else:
            self.m_head = None

        # initialize gamma small (helps stability)
        nn.init.normal_(self.gamma.weight, mean=0.0, std=0.02)

    def extract_features(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        return x  # (n, d)

    def forward(self, x, a):
        """
        x: (n, p)
        a: (n,) long tensor with values in {0,...,K-1}
        returns y_hat: (n,) float tensor
        """
        h = self.extract_features(x)                       # (n, d)
        g = self.gamma(a.long())                           # (n, d)

        # interaction term <h, g>
        inter = torch.sum(h * g, dim=1)                    # (n,)

        if self.m_head is not None:
            base = self.m_head(h).squeeze(1)              # (n,)
            y_hat = base + inter
        else:
            y_hat = inter

        return y_hat

    def predict_from_HA(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        g = self.gamma(A.long())  # (n, d)
        inter = torch.sum(H * g, dim=1)  # (n,)

        if self.m_head is not None:
            base = self.m_head(H).squeeze(1)  # (n,)
            return base + inter
        return inter

    def gamma_matrix(self):
        """Return gamma as a (K, d) Parameter-like tensor for ADMM."""
        return self.gamma.weight
