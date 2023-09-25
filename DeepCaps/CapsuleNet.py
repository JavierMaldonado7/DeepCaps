import torch
import torch.nn.functional as F

import Routing
from Decoder import CapsuleDecoder
from Encoder import Encoder


class CapsuleNet(torch.nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=2):
        super(CapsuleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9)
        self.primary_capsules = PrimaryCapsules()
        self.digit_capsules = DigitCapsules()
        self.decoder = CapsuleDecoder(num_output_capsules=10, output_cap_channels=16, reconstruction_input_size=32 * 32)
        self.encoder = Encoder(input_channels=3, conv_channels=256, num_primary_capsules= 9,primary_cap_channels= 32 * 8, num_output_capsules= 32 *8, output_cap_channels= 32 * 8);

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x


class PrimaryCapsules(torch.nn.Module):
    def __init__(self):
        super(PrimaryCapsules, self).__init__()
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2)

    def forward(self, x):
        x = self.conv2(x)
        x = x.view(x.size(0), 32 * 8, -1)
        x = self.squash(x)
        return x

    def squash(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        x = scale * (x / norm)
        return x


class DigitCapsules(torch.nn.Module):
    def __init__(self, num_output_capsules=2, output_cap_channels=16, num_iterations=3, routing_fn='EMRouting'):
        super(DigitCapsules, self).__init__()
        self.num_output_capsules = num_output_capsules
        self.output_cap_channels = output_cap_channels
        self.num_iterations = num_iterations

        self.routing_fn = Routing.Routing

        self.capsules = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=256,
                    out_channels=output_cap_channels,
                    kernel_size=1,
                    stride=1
                ),
                torch.nn.Flatten()
            )
            for _ in range(num_output_capsules)
        ])

        self.weights = torch.nn.Parameter(
            torch.randn(num_output_capsules, 16, 8, output_cap_channels)
        )

    def forward(self, x):
        """
        :param x: input tensor of size [batch_size, 256, 6, 6]
        :return: output tensor of size [batch_size, num_output_capsules, 16]
        """
        batch_size = x.size(0)
        caps_out = [caps(x).unsqueeze(1) for caps in self.capsules]  # shape: [batch_size, 1, output_cap_channels]
        caps_out = torch.cat(caps_out, dim=1)  # shape: [batch_size, num_output_capsules, output_cap_channels]

        # tile the weights for each batch sample
        weights = self.weights.unsqueeze(0).repeat(batch_size, 1, 1, 1,
                                                   1)  # shape: [batch_size, num_output_capsules, 16, output_cap_channels]

        # calculate the agreement between capsules
        u = (caps_out.unsqueeze(2) @ weights).squeeze(
            3)  # shape: [batch_size, num_output_capsules, 16, output_cap_channels]

        # perform routing to obtain the output capsules
        v = self.routing_fn(u)

        return v
