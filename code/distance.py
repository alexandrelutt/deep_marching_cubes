import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from distance_extension import distance_cuda

class DistPtsTopoFct(Function):
    @staticmethod
    def forward(ctx, offset, points):
        N = offset.size(1)
        T = 48
        distances_full = torch.zeros(((N-1)**3, T), dtype=offset.dtype, device=offset.device)
        indices = -1*torch.ones((points.size(0), T), dtype=torch.long, device=offset.device)
        distance_cuda.pt_topo_distance_cuda_forward(offset, points, distances_full, indices)
        ctx.save_for_backward(offset, points)
        ctx.indices = indices
        return distances_full

    @staticmethod
    def backward(ctx, grad_output):
        offset, points = ctx.saved_tensors
        indices = ctx.indices
        grad_offset = torch.zeros_like(offset).type(torch.FloatTensor).cuda()
        distance_cuda.pt_topo_distance_cuda_backward(grad_output, offset, points, indices, grad_offset)
        return grad_offset, None

class DistPtsTopo(torch.nn.Module):
    def forward(self, offset, points):
        return DistPtsTopoFct.apply(offset, points)