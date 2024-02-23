import numpy as np
import torch
import time

from code.model import RawDeepMarchingCube
from code.my_losses import PointToMeshLoss

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_data = np.load('all_data/points_shapenet_32x32x32_train.npy')
    batch_size = 8

    sample_batch = train_data[:batch_size, :, :]
    sample_batch = torch.tensor(sample_batch).float().to(device)

    model = RawDeepMarchingCube()
    model.to(device)
    
    print(f'Computing DMC...')
    t = time.time()
    offset, topology, occupancy = model(sample_batch)
    dt = time.time() - t
    print(f'Finished computing DMC in {dt:.4f} seconds.')

    _fake_offset = torch.zeros_like(offset).sum(1).sum(1).sum(1)
    _offset = offset.sum(1).sum(1).sum(1)
        
    criterion = PointToMeshLoss()
    print(f'Computing loss...')
    t = time.time()
    loss_offset = criterion(_fake_offset, _offset)
    dt = time.time() - t
    print(f'Finished computing loss in {dt:.4f} seconds.')

    print(f'Backward...')
    t = time.time()
    loss_offset.backward()
    dt = time.time() - t
    print(f'Finished backward in {dt:.4f} seconds.')
    print(f'Finished.')
