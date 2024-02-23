from torch.utils.cpp_extension import load

GPooling = load(name="grid_pooling", sources=["../cpp_files/grid_pooling.cpp"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    def __init__(self, n_features_in=3, n_features_hid=256, n_features_out=16):
        super(FeatureExtractor, self).__init__()

        self.n_features_in = n_features_in
        self.n_features_hid = n_features_hid
        self.n_features_out = n_features_out

        self.conv1 = torch.nn.Conv1d(n_features_in, n_features_hid, 1)
        self.conv2 = torch.nn.Conv1d(n_features_hid, n_features_out, 1)

    def forward(self, point_cloud):
        point_cloud = point_cloud.transpose(2, 1)
        point_cloud = F.relu(self.conv1(point_cloud))
        features = self.conv2(point_cloud)
        features = features.transpose(1, 2)
        return features
    
class GridPooling(Function):
    @staticmethod
    def forward(ctx, point_cloud, features):
        N = 32

        batch_size = point_cloud.shape[0]
        C = features.shape[-1]

        volumetric_representation = torch.zeros((batch_size, N, N, N, C))
        indices = -1*torch.ones((batch_size, N**3, C), dtype=torch.int)

        GPooling.grid_pooling(volumetric_representation, indices, point_cloud, features)

        ctx.save_for_backward(indices)
        ctx.n_points = point_cloud.shape[1]
        ctx.N = N   
        ctx.C = C
        ctx.batch_size = batch_size

        return volumetric_representation
    
    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]
        n_points = ctx.n_points
        N = ctx.N
        C = ctx.C
        batch_size = ctx.batch_size

        grad_points = torch.zeros((batch_size, n_points, C))
        GPooling.grid_pooling_backward(grad_points, grad_output, indices, N)

        return grad_points, None, None