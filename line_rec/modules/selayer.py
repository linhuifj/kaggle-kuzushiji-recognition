import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, in_channels, reduction = 16):
        super(SELayer, self).__init__()
        mid_channels = int(in_channels / reduction)
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        n_baches, n_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, output_size = 1).view(n_baches, n_channels)
        y = F.relu(self.fc1(y), inplace = True)
        y = F.sigmoid(self.fc2(y)).view(n_baches, n_channels, 1,1)
        return x*y

                      
