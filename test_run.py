import numpy as np
import torch
import time

from code.model import DeepMarchingCube
from code.loss import MyLoss
from code.loader import get_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 8
    train_loader = get_loader(set='train', batch_size=batch_size)

    model = DeepMarchingCube()
    model.to(device)

    for i, (clean_batch, perturbed_batch) in enumerate(train_loader):
        clean_batch = clean_batch.to(device)
        perturbed_batch = perturbed_batch.to(device)

        print(f'Performing forward pass...')
        t = time.time()
        offset, topology, occupancy = model(perturbed_batch)
        dt = time.time() - t
        print(f'Finished computing full forward pass in {dt:.4f} seconds.')

        loss = MyLoss().point_to_mesh(offset, topology, clean_batch, occupancy)

        print(f'Performing backward pass...')
        t = time.time()
        loss.backward()
        dt = time.time() - t
        print(f'Finished computing full backward pass in {dt:.4f} seconds.')

        break