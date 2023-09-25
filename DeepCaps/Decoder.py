import torch


class CapsuleDecoder(torch.nn.Module):
    def __init__(self, num_output_capsules, output_cap_channels, reconstruction_input_size):
        super(CapsuleDecoder, self).__init__()
        self.num_output_capsules = num_output_capsules
        self.output_cap_channels = output_cap_channels
        self.reconstruction_input_size = reconstruction_input_size

        self.reconstruction_layers = torch.nn.Sequential(
            torch.nn.Linear(num_output_capsules * output_cap_channels, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, reconstruction_input_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x, target):
        batch_size = x.size(0)
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        _, index = v_c.max(dim=1)
        index = index.data

        if torch.cuda.is_available():
            one_hot = torch.zeros(batch_size, self.num_output_capsules).cuda()
        else:
            one_hot = torch.zeros(batch_size, self.num_output_capsules)

        one_hot.scatter_(1, index.view(-1, 1), 1.)
        x = (x * one_hot[:, :, None]).view(x.size(0), -1)
        x = self.reconstruction_layers(x)
        target = target.view(-1, self.reconstruction_input_size)
        return x, target
