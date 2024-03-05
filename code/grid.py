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
        
class GridPoolingFct(Function):
    @staticmethod
    def forward(ctx, points, features):
        batch_size, n_points, _ = points.size()
        _, _, n_features = features.size()

        output_grid = torch.zeros(batch_size, 32, 32, 32, n_features, device=points.device)

        indices = -1*torch.ones(batch_size, 32**3, n_features, device=points.device, dtype=torch.long)

        normed_points = (points - points.min()) / (points.max() - points.min() + 1e-6)
        voxel_indices = (normed_points // (1/32)).long()
        print(voxel_indices.min())
        print(voxel_indices.max())

        for b in range(batch_size):
            for i in range(n_points):
                voxel_index = voxel_indices[b, i]
                global_index = voxel_index[0]*32*32 + voxel_index[1]*32 + voxel_index[2]

                global_t = torch.cat([output_grid[b, voxel_index[0], voxel_index[1], voxel_index[2]].reshape(n_features, -1), features[b, i].reshape(n_features, -1)], dim=1)
                max_values, max_indices = torch.max(global_t, dim=1)

                output_grid[b, voxel_index[0], voxel_index[1], voxel_index[2]] = max_values

                old_indices = indices[b, global_index]
                indices[b, global_index] = torch.where(max_indices.bool(), global_index, old_indices)
                
        ctx.save_for_backward(indices)
        ctx.batch_size = batch_size
        ctx.n_points = n_points
        ctx.n_features = n_features
        return output_grid

    @staticmethod
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors[0]
        batch_size = ctx.batch_size
        n_points = ctx.n_points
        n_features = ctx.n_features

        grad_points = torch.zeros((batch_size, n_points, n_features))
        for b in range(batch_size):
            for i in range(n_points):
                for c in range(n_features):
                    current_indices = indices[b, i, c]
                    if current_indices == -1:
                        continue
                    grad_points[b, i, c] = grad_output[b, current_indices//32//32, (current_indices//32)%32, current_indices%32, c]
        return grad_points, None

class GridPooling(nn.Module):
    def forward(self, points, features):
        return GridPoolingFct.apply(points, features)