import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    def __init__(self, num_features, hid_dim, out_dim, return_hidden=False):
        super().__init__()
        self.return_hidden = return_hidden
        self.model = nn.Sequential(
            nn.Linear(num_features, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not self.return_hidden:
            return self.model(x)
        else:
            hid_feat = self.model[:2](x)
            results = self.model[2:](hid_feat)
            return hid_feat, results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

