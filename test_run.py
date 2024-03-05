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

    batch_size = 8
    n_epochs = 10
    learning_rate = 5e-4

    train_loader = get_loader(set='train', batch_size=batch_size)
    test_loader = get_loader(set='test', batch_size=batch_size)

    model = DeepMarchingCube()
    loss_module = MyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = train(model, train_loader, test_loader, loss_module, n_epochs, optimizer, device)

    print('Train losses:')
    print(train_losses)

    print('Test losses:')
    print(test_losses)