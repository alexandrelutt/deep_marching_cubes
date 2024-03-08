import torch

class Smoothness(torch.nn.Module):
    def __init__(self):
        super(Smoothness, self).__init__()
        
    def forward(self, occupancy):    
        diff_x = occupancy[:-1, :, :] - occupancy[1:, :, :]
        diff_y = occupancy[:, :-1, :] - occupancy[:, 1:, :]
        diff_z = occupancy[:, :, :-1] - occupancy[:, :, 1:]
        
        loss = torch.sum(torch.abs(diff_x)) + torch.sum(torch.abs(diff_y)) + torch.sum(torch.abs(diff_z))
        print(loss, type(loss))
        return loss