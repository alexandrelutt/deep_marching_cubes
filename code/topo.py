from torch.utils.cpp_extension import load

OccTopology = load(name="occtopology", sources=["../cpp_files/occtopology.cpp"])

import torch
from torch.autograd import Function

class OccupancyToTopology(Function):
    @staticmethod
    def forward(ctx, occupancy):
        batch_size = occupancy.shape[0]
        N = occupancy.shape[2]-1
        T = 256
        topology = torch.zeros(batch_size, N**3, T).float() 
        for b in range(batch_size):
            OccTopology.occupancy_to_topology(occupancy[b, 0, :, :, :], topology[b])

        ctx.save_for_backward(occupancy)
        ctx.batch_topology = topology

        return topology

    @staticmethod
    def backward(ctx, grad_output):
        occupancy = ctx.saved_tensors[0]
        topology = ctx.batch_topology

        batch_size = occupancy.shape[0]
        grad_occupancy = torch.zeros_like(occupancy).float()
        for b in range(batch_size):
            OccTopology.occupancy_to_topology_backward(grad_output[b], occupancy[b, 0, :, :, :], topology[b], grad_occupancy[b, 0, :, :, :])

        return grad_occupancy