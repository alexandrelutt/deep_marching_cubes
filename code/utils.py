import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, loss_type='global', log=False):
    n_epochs = len(train_losses)
    fig = plt.figure(figsize=(10, 5))

    if log:
        plt.plot(range(1, 1+n_epochs), np.log(train_losses), label='Train loss')
        plt.plot(range(1, 1+n_epochs), np.log(test_losses), label='Test loss')
    else:
        plt.plot(range(1, 1+n_epochs), train_losses, label='Train loss')
        plt.plot(range(1, 1+n_epochs), test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    str_title = f'Train and test {loss_type} loss'
    if log:
        str_title += ' (log scale)'

    plt.legend()  
    if log:
        plt.savefig(f'outputs/figures/{loss_type}_training_log.png')
    else:
        plt.savefig(f'outputs/figures/{loss_type}_training.png')
    plt.close()