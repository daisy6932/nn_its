# Archive_new/model.py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features, n_classes, hidden1=32, hidden2=16, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(hidden2, n_classes, bias=False)

    def extract_features(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        return x

    def forward(self, x):
        h = self.extract_features(x)
        return self.output_layer(h)

