import numpy as np
import torch
import time
from tqdm import tqdm
import faulthandler

from code.model import DeepMarchingCube
from code.loss import MyLoss
from code.loader import get_loader
from code.train import train

if __name__ == '__main__':
    faulthandler.enable()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 8
    train_loader = get_loader(set='train', batch_size=batch_size)
    test_loader = get_loader(set='test', batch_size=batch_size)

    model = DeepMarchingCube()
    loss_module = MyLoss()
    n_epochs = 10

    train_losses, test_losses = train(model, train_loader, test_loader, loss_module, n_epochs, device)
    print('Training complete')

    print('Train losses:')
    print(train_losses)

    print('Test losses:')
    print(test_losses)