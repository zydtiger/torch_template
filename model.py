import torch.nn as nn


class LogisticMLP(nn.Module):
    def __init__(self, input_dim=5):
        super(LogisticMLP, self).__init__()
        self.input = nn.Linear(input_dim, 32)
        self.reduce_dim1 = nn.Linear(32, 16)
        self.reduce_dim2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.dropout(x)
        x = self.relu(self.reduce_dim1(x))
        x = self.dropout(x)
        x = self.reduce_dim2(x)
        x = self.output(x)
        return x
