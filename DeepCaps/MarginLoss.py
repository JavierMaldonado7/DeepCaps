import torch
import torch.optim as optim
import torch.nn.functional as F
# Define loss function
from torch import nn


class MarginLoss(nn.Module):
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, output, target):
        # Convert target to one-hot encoding
        target_onehot = torch.eye(2)[target]

        # Compute lengths of output vectors
        lengths = torch.sqrt((output ** 2).sum(dim=2))

        # Compute margin loss
        loss_pos = target_onehot * F.relu(self.m_pos - lengths).pow(2)
        loss_neg = (1 - target_onehot) * F.relu(lengths - self.m_neg).pow(2)
        margin_loss = loss_pos + self.lambda_ * loss_neg
        margin_loss = margin_loss.sum(dim=1).mean()

        return margin_loss