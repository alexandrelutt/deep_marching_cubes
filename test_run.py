import numpy as np
import torch
import time
import faulthandler

from code.model import DeepMarchingCube
# from code.loss import MyLoss
from code.loader import get_loader

if __name__ == '__main__':
    faulthandler.enable()
    
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

        print('')
        print(f'Offset shape: {offset.shape}')
        print(f'Topology shape: {topology.shape}')
        print(f'Occupancy shape: {occupancy.shape}')

        # print(f'Computing loss...')
        # t = time.time()
        # loss = MyLoss().loss(offset, topology, clean_batch, occupancy)
        # dt = time.time() - t
        # print(f'Finished computing loss in {dt:.4f} seconds.')

        # print(f'Performing backward pass...')
        # t = time.time()
        # loss.backward()
        # dt = time.time() - t
        # print(f'Finished computing full backward pass in {dt:.4f} seconds.')

        break