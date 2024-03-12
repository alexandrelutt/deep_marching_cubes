import torch
import pickle
import faulthandler

from code.model import DeepMarchingCube
from code.loss import MyLoss
from code.loader import get_loader
from code.train import train

if __name__ == '__main__':
    faulthandler.enable()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 8
    n_epochs = 100 # base 500
    learning_rate = 5e-5 ## base 5e-4
    weight_decay = 1e-3

    train_loader = get_loader(set='train', batch_size=batch_size)
    test_loader = get_loader(set='test', batch_size=batch_size)

    model = DeepMarchingCube()
    loss_module = MyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses_dict, test_losses_dict, best_model = train(model, train_loader, test_loader, loss_module, n_epochs, optimizer, scheduler, device)

    with open('train_losses_dict.pkl', 'wb') as f:
        pickle.dump(train_losses_dict, f)
    with open('test_losses_dict.pkl', 'wb') as f:
        pickle.dump(test_losses_dict, f)