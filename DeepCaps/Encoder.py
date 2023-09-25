import torch
import torch.nn as nn
import torch.nn.functional as F

from Routing import Routing


class Encoder(nn.Module):
    def __init__(self, input_channels, conv_channels, num_primary_capsules, primary_cap_channels, num_output_capsules, output_cap_channels):
        super(Encoder, self).__init__()
        self.num_primary_capsules = num_primary_capsules
        self.primary_cap_channels = primary_cap_channels
        self.num_output_capsules = num_output_capsules
        self.output_cap_channels = output_cap_channels

        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.primary_capsules = nn.ModuleList([
            nn.Conv2d(conv_channels, primary_cap_channels, kernel_size=1, stride=1, padding=0) for _ in range(num_primary_capsules)])
        self.routing = Routing()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        u = [capsule(x) for capsule in self.primary_capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_primary_capsules, -1)
        a = self.routing(u)
        return a