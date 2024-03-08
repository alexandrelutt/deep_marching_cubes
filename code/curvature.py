import torch
from torch.autograd import Function
from code.table import get_connected_pairs

from curvature_extension import curvature_cuda

x, y, z, inner, topology_to_triangles = get_connected_pairs()

class CurvatureFct(Function):
    @staticmethod
    def forward(ctx, offset, topology):
        loss = torch.zeros(1).type(torch.FloatTensor).cuda()
        curvature_cuda.curvature_constraint_cuda_forward(
			    offset,
			    topology[:, torch.LongTensor(topology_to_triangles).cuda()],
			    # topology[:, -1],
			    torch.FloatTensor(x).cuda(),
			    torch.FloatTensor(y).cuda(),
			    torch.FloatTensor(z).cuda(),
			    torch.FloatTensor(inner).cuda(),
			    loss)
        ctx.save_for_backward(offset, topology)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        offset, topology = ctx.saved_tensors
        grad_offset = torch.zeros(offset.size()).cuda()
        curvature_cuda.curvature_constraint_cuda_backward(
            grad_output,
		    offset,
		    topology[:, torch.LongTensor(topology_to_triangles).cuda()],
		    # topology[:, -1],			    
            torch.FloatTensor(x).cuda(),
			torch.FloatTensor(y).cuda(),
		    torch.FloatTensor(z).cuda(),
		    torch.FloatTensor(inner).cuda(),
		    grad_offset)
        grad_topology = torch.zeros(topology.size()).cuda()
        return grad_offset, grad_topology 


class Curvature(torch.nn.Module):
    def forward(self, offset, topology):
        loss = CurvatureFct.apply(offset, topology)
        return loss