import numpy as np
import torch
import time

from code.model import DeepMarchingCube

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_data = np.load('all_data/points_shapenet_32x32x32_train.npy')
    batch_size = 8

    sample_batch = train_data[:batch_size, :, :]
    sample_batch = torch.tensor(sample_batch).float().to(device)

    model = DeepMarchingCube()
    model.to(device)
    
    print(f'Performing forward pass...')
    t = time.time()
    offset, topology, occupancy = model(sample_batch)
    dt = time.time() - t
    print(f'Finished computing full forward pass in {dt:.4f} seconds.')

    fake_topology = torch.zeros_like(topology)
    loss = ((fake_topology - topology)**2).sum()

    print(f'Performing backward pass...')
    t = time.time()
    loss.backward()
    dt = time.time() - t
    print(f'Finished computing full backward pass in {dt:.4f} seconds.')