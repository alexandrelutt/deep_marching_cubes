import torch
import numpy as np
import scipy
import torch.nn.functional as F

from code.distance import DistPtsTopo
from code.smoothness import Smoothness
from code.curvature import Curvature
from code.table import get_accepted_topologies

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

def get_gaussian_kernel_3D(kernel_width, sigma=1):
    kernel_side = np.arange(-kernel_width//2 + 1, kernel_width//2 + 1)
    x, y, z = np.meshgrid(kernel_side, kernel_side, kernel_side)
    kernel = np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2))
    return kernel

class MyLoss(object):
    def __init__(self):
        self.N = 32

        self.weight_point_to_mesh = 1.0 
        self.weight_occupancy = 0.4
        self.weight_smoothness = 0.6
        self.weight_curvature = 0.6

        ## utils for point_to_mesh loss
        self.dist_pt_topo = DistPtsTopo()

        self.acceptedTopologies = torch.Tensor(get_accepted_topologies()).type(torch.LongTensor).cuda()
        indices = torch.arange(self.acceptedTopologies.size()[0]-1, -1, -1).type(torch.IntTensor)
        self.acceptTopologyWithFlip = torch.cat([self.acceptedTopologies, 255-self.acceptedTopologies[indices]], dim=0)

        ## utils for occupancy loss
        cube_boundaries = np.zeros((self.N+1, self.N+1, self.N+1))
        cube_boundaries[0, :, :] = 1
        cube_boundaries[self.N, :, :] = 1
        cube_boundaries[:, :, 0] = 1
        cube_boundaries[:, :, self.N] = 1
        cube_boundaries[:, 0, :] = 1
        cube_boundaries[:, self.N, :] = 1

        gaussian_kernel_3D = get_gaussian_kernel_3D(kernel_width=3)
        neg_weight = scipy.ndimage.filters.convolve(cube_boundaries, gaussian_kernel_3D)
        neg_weight = neg_weight/np.max(neg_weight)
        self.neg_weight = torch.from_numpy(neg_weight).type(dtype)

        self.fraction_inside = 0.2

        ## utils for smoothness loss
        self.smoothness = Smoothness()

        ## utils for curvature loss
        self.curvature = Curvature()

    def point_to_mesh(self, offset, topology, pts):
        distances_point_to_topo = self.dist_pt_topo(offset, pts)

        indices = torch.arange(len(self.acceptedTopologies)-1, -1, -1).type(torch.IntTensor)
        accepted_dists = torch.cat([distances_point_to_topo, distances_point_to_topo[:, indices]], dim=1)

        accepted_topos = topology[:, self.acceptTopologyWithFlip]
        probas_sum = torch.sum(accepted_topos, dim=1, keepdim=True).clamp(min=1e-6)
        accepted_topos = accepted_topos/probas_sum
        loss = torch.sum(accepted_topos.mul(accepted_dists))/self.N**3
        return loss

    def occupancy_loss(self, occupancy):
        loss_sides = torch.sum(torch.mul(occupancy, self.neg_weight)) / torch.sum(self.neg_weight)

        N = occupancy.size(0)
        sorted_cube,_ = torch.sort(occupancy.detach().view(-1), 0, descending=True)
        treshold = int(sorted_cube.size(0)/30)
        ## mean of the 30% highest values
        weight_to_adapt = 1 - torch.mean(sorted_cube[:treshold])
        min_bound, max_bound = int(0.2*N), int(0.8*N)
        loss_inside = 1 - torch.mean(occupancy[min_bound:max_bound, min_bound:max_bound, min_bound:max_bound])
        loss_inside = self.fraction_inside*weight_to_adapt*loss_inside

        loss = loss_sides + loss_inside
        return loss/self.N**3

    def smoothness_loss(self, occupancy):
        loss = self.smoothness(occupancy)/self.N**3
        return loss
    
    def curvature_loss(self, offset, topology):
        topology_accepted = topology[:, self.acceptTopologyWithFlip]
        loss = self.curvature(offset, F.softmax(topology_accepted, dim=1))[0]/self.N**3
        return loss
    
    def loss(self, offset, topology, pts, occupancy):
        batch_size = offset.size(0)
        loss_point_to_mesh = 0
        loss_occupancy = 0
        loss_smoothness = 0
        loss_curvature = 0

        for b in range(batch_size):
            loss_point_to_mesh += self.weight_point_to_mesh*self.point_to_mesh(offset[b], topology[b], pts[b])
            loss_occupancy += self.weight_occupancy*self.occupancy_loss(occupancy[b, 0])
            loss_smoothness += self.weight_smoothness*self.smoothness_loss(occupancy[b, 0])
            loss_curvature += self.weight_curvature*self.curvature_loss(offset[b], topology[b])
            
        loss = loss_point_to_mesh + loss_occupancy + loss_smoothness + loss_curvature
        return loss/batch_size, loss_point_to_mesh/batch_size, loss_occupancy/batch_size, loss_smoothness/batch_size, loss_curvature/batch_size