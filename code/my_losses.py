import torch
import torch.nn as nn

class PointToMeshLoss(nn.modules.loss._Loss):
        def __init__(self):
            super(PointToMeshLoss, self).__init__()

        def forward(self, input, target):
            return torch.nn.MSELoss()(input, target)