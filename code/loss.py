import torch
import torch.nn as nn
import torch.nn.functional as F

from code.distance import DistPtsTopo
from code.table import get_accepted_topologies

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

class MyLoss(object):
    def __init__(self):
        self.N = 32
        self.weight_point_to_mesh = 1/self.N**3
        self.weight_occupancy = 0.4/self.N**3
        self.weight_smoothness = 0.6/self.N**3
        self.weight_curvature = 0.6/self.N**3

        self.dist_pt_topo = DistPtsTopo()

        self.acceptedTopologies = torch.Tensor(get_accepted_topologies()).type(torch.LongTensor).cuda()
        indices = torch.arange(self.acceptedTopologies.size()[0]-1, -1, -1).type(torch.LongTensor)
        self.acceptTopologyWithFlip = torch.cat([self.acceptedTopologies, 255-self.acceptedTopologies[indices]], dim=0)

    def point_to_mesh(self, offset, topology, pts, occupancy):
        distances_point_to_topo = self.distances(offset, pts)

        indices = torch.arange(len(self.acceptedTopologies)-1, -1, -1).type(torch.LongTensor)
        accepted_dists = torch.cat([distances_point_to_topo, distances_point_to_topo[:, indices]], dim=1)

        accepted_topos = topology[:, self.acceptTopologyWithFlip]
        probas_sum = torch.sum(accepted_topos, dim=1, keepdim=True).clamp(min=1e-6)
        accepted_topos = accepted_topos/probas_sum
        loss = torch.sum(accepted_topos.mul(accepted_dists))/self.N**3
        return loss
    
    def occupancy_loss(self, occupancy):
        raise NotImplementedError

    def smoothness_loss(self, occupancy):
        raise NotImplementedError

    def curvature_loss(self, offset, topology):
        raise NotImplementedError
    
    def loss(self, offset, topology, pts, occupancy):
        batch_size = offset.size(0)
        loss = 0

        for b in range(batch_size):
            loss += self.weight_point_to_mesh*self.point_to_mesh(offset[b], topology[b], pts[b])
            # loss += self.weight_occupancy*self.occupancy_loss(occupancy[b])
            # loss += self.weight_smoothness*self.smoothness_loss(occupancy[b])
            # loss += self.weight_curvature*self.curvature_loss(offset[b], topology[b])

        return loss/batch_size