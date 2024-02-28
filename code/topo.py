import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.cpp_extension import load

# OccTopology = load(name="occtopology", sources=["cpp_files/occtopology_cuda.cpp", "cpp_files/occtopology_cuda_kernel.cu"])
# OccTopology = load(name="occtopology", sources=["occtopology_cuda.cpp", "occtopology_cuda_kernel.cu"])

from dist import cuda_extension.py

class OccupancyToTopologyFct(Function):
    @staticmethod
    def forward(ctx, occupancy):
        N = occupancy.size(0) - 1
        T = 256
        topology = torch.zeros(N**3, T, device=occupancy.device).float()
        OccTopology.occ_to_topo_cuda_forward(occupancy, topology)

        ctx.save_for_backward(occupancy, topology)
        return topology
    
    @staticmethod
    def backward(ctx, grad_output):
        occupancy, topology = ctx.saved_tensors
        grad_occupancy = torch.zeros_like(occupancy).type(torch.FloatTensor).cuda()
        OccTopology.occ_to_topo_cuda_backward(grad_output, occupancy, topology, grad_occupancy)
        return grad_occupancy 
    
class OccupancyToTopology(nn.Module):
    def forward(self, occupancy):
        return OccupancyToTopologyFct.apply(occupancy)