import torch
import faulthandler

from code.model import DeepMarchingCube
from code.loss import MyLoss
from code.loader import get_loader
from code.train import train

if __name__ == '__main__':
    faulthandler.enable()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    n_epochs = 100
    learning_rate = 5e-4
    weight_decay = 1e-3

    train_loader = get_loader(set='train', batch_size=batch_size)
    test_loader = get_loader(set='test', batch_size=batch_size)

    model = DeepMarchingCube()
    loss_module = MyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(model, train_loader, test_loader, loss_module, n_epochs, optimizer, device)