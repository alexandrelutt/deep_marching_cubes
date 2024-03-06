import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from smoothness_extension import smoothness_cuda

class SmoothnessFct(Function):
    @staticmethod
    def forward(ctx, occupancy):
        loss = torch.zeros(1).type(torch.FloatTensor).cuda()
        smoothness_cuda.connectivity_cuda_forward(occupancy, loss)
        print(f'Loss: {loss.item()}')
        ctx.save_for_backward(occupancy)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        occupancy = ctx.saved_tensors
        grad_occupancy = torch.zeros(occupancy.size()).type(torch.FloatTensor).cuda()
        smoothness_cuda.connectivity_cuda_backward(grad_output, occupancy, grad_occupancy)
        return grad_occupancy


class Smoothness(torch.nn.Module):
    def forward(self, occupancy):
        return SmoothnessFct.apply(occupancy)