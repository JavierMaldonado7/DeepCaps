import torch


class Routing(torch.nn.Module):
    def __init__(self, num_iterations=3):
        super(Routing, self).__init__()
        self.num_iterations = num_iterations

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        batch_size = x.size(0)
        num_input_capsules = x.size(1)
        num_output_capsules = self.weight.size(0)
        out_dim = self.weight.size(2)
        in_dim = self.weight.size(3)

        x = x.unsqueeze(2).unsqueeze(4)
        weight = self.weight.unsqueeze(0).unsqueeze(4).repeat(batch_size, 1, 1, out_dim, 1, 1)
        u_hat = torch.matmul(weight, x)
        u_hat = u_hat.squeeze(-1)
        b = torch.zeros(batch_size, num_output_capsules, num_input_capsules, 1).to(x.device)
        for i in range(self.num_iterations):
            c = torch.nn.functional.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2, keepdim=True)
            v = self.squash(s)
            b = b + (u_hat * v).sum(dim=-1, keepdim=True)
        return v.squeeze(2)
