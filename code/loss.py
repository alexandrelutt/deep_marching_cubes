import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

class MyLoss(object):
    def __init__(self):
        pass

    def point_to_mesh(self, offset, topology, pts, occupancy):
        pass