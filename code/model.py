import torch
import torch.nn as nn
import torch.nn.functional as F

from code.grid import GridPooling, FeatureExtractor
from code.topo import OccupancyToTopology

# from grid import GridPooling, FeatureExtractor
# from topo import OccupancyToTopology

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

class LocalEncoder(nn.Module):
    """Encoder of the U-Net"""
    def __init__(self, input_dim=16):
        super(LocalEncoder, self).__init__()

        self.conv1_1 = nn.Conv3d(input_dim, 16, 3, padding=3)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)

        self.conv1_1_bn = nn.BatchNorm3d(16)
        self.conv1_2_bn = nn.BatchNorm3d(16)
        self.conv2_1_bn = nn.BatchNorm3d(32)
        self.conv2_2_bn = nn.BatchNorm3d(32)
        self.conv3_1_bn = nn.BatchNorm3d(64)
        self.conv3_2_bn = nn.BatchNorm3d(64)
        self.conv4_bn   = nn.BatchNorm3d(128)

        self.maxpool = nn.MaxPool3d(2, return_indices=True)

    def encoder(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x)))
        feat1 = x
        size1 = x.size()
        x, indices1 = self.maxpool(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        feat2 = x
        size2 = x.size()
        x, indices2 = self.maxpool(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        feat3 = x
        size3 = x.size()
        x, indices3 = self.maxpool(x)

        x = F.relu(self.conv4_bn(self.conv4(x)))
        return x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3
    
    def forward(self, x):
        x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = self.encoder(x)
        return x, (feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3)
        
class SurfaceDecoder(nn.Module):
    """Decoder of the U-Net, estimate topology and offset with two headers"""
    def __init__(self):
        super(SurfaceDecoder, self).__init__()

        self.deconv4 = nn.Conv3d(128, 64, 3, padding=1)
        self.deconv3_1 = nn.ConvTranspose3d(128, 128, 3, padding=1)
        self.deconv3_2 = nn.ConvTranspose3d(128, 32, 3, padding=1)
        self.deconv2_off_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_off_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv2_occ_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_occ_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv1_off_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_off_2 = nn.ConvTranspose3d(32, 3 , 3, padding=3)
        self.deconv1_occ_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_occ_2 = nn.ConvTranspose3d(32, 1 , 3, padding=3)

        self.deconv4_bn = nn.BatchNorm3d(64)
        self.deconv3_1_bn = nn.BatchNorm3d(128)
        self.deconv3_2_bn = nn.BatchNorm3d(32)
        self.deconv2_off_1_bn = nn.BatchNorm3d(64)
        self.deconv2_off_2_bn = nn.BatchNorm3d(16)
        self.deconv2_occ_1_bn = nn.BatchNorm3d(64)
        self.deconv2_occ_2_bn = nn.BatchNorm3d(16)
        self.deconv1_off_1_bn = nn.BatchNorm3d(32)
        self.deconv1_occ_1_bn = nn.BatchNorm3d(32)

        self.sigmoid = nn.Sigmoid()

        self.maxunpool = nn.MaxUnpool3d(2)

    def decoder(self, x, intermediate_feat=None):

        if self.skip_connection:
            feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = intermediate_feat

        x = F.relu(self.deconv4_bn(self.deconv4(x)))

        x = self.maxunpool(x, indices3, output_size=size3)
        x = torch.cat((feat3, x), 1)
        x = F.relu(self.deconv3_1_bn(self.deconv3_1(x)))
        x = F.relu(self.deconv3_2_bn(self.deconv3_2(x)))

        x = self.maxunpool(x, indices2, output_size=size2)
        x = torch.cat((feat2, x), 1)
        x_occupancy = F.relu(self.deconv2_occ_1_bn(self.deconv2_occ_1(x)))
        x_occupancy = F.relu(self.deconv2_occ_2_bn(self.deconv2_occ_2(x_occupancy)))
        x_offset = F.relu(self.deconv2_off_1_bn(self.deconv2_off_1(x)))
        x_offset = F.relu(self.deconv2_off_2_bn(self.deconv2_off_2(x_offset)))

        x_occupancy = self.maxunpool(x_occupancy, indices1, output_size=size1)
        x_occupancy = torch.cat((feat1, x_occupancy), 1)
        x_offset = self.maxunpool(x_offset, indices1, output_size=size1)
        x_offset = torch.cat((feat1, x_offset), 1)
        x_occupancy = F.relu(self.deconv1_occ_1_bn(self.deconv1_occ_1(x_occupancy)))
        x_occupancy = self.sigmoid(self.deconv1_occ_2(x_occupancy))
        x_offset = F.relu(self.deconv1_off_1_bn(self.deconv1_off_1(x_offset)))
        x_offset = self.sigmoid(self.deconv1_off_2(x_offset)) - 0.5

        return x_occupancy, x_offset

    def forward(self, x, intermediate_feat=None):
        return self.decoder(x, intermediate_feat)
    
class DeepMarchingCube(nn.Module):
    def __init__(self):
        super(DeepMarchingCube, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.grid_pooling = GridPooling()
        self.encoder = LocalEncoder(16)
        self.decoder = SurfaceDecoder()
        self.occupancy_to_topology = OccupancyToTopology()

        self.N = 32

    def forward(self, x):
        features = self.feature_extractor(x)

        output_grid = self.grid_pooling(x, features)
        output_grid = output_grid.permute(0, 4, 1, 2, 3)
        
        curr_size = output_grid.size()
        new_output_grid = torch.zeros(curr_size[0], curr_size[1], curr_size[2]+1, curr_size[3]+1, curr_size[4]+1)
        new_output_grid[:, :, :-1, :-1, :-1] = output_grid
        if next(self.parameters()).is_cuda:
            new_output_grid = new_output_grid.cuda()

        x, intermediate_feat = self.encoder(new_output_grid)
        occupancy, offset = self.decoder(x, intermediate_feat)
        
        N = occupancy.size(2) - 1
        T = 256
        batch_size = occupancy.size(0)
        topology = torch.zeros(batch_size, N**3, T, device=occupancy.device).float()
        
        for b in range(batch_size):
            topology[b, :, :] = self.occupancy_to_topology(occupancy[b, 0, :, :, :])

        return offset, topology, occupancy