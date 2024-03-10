import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, loss_type='all'):
    n_epochs = len(train_losses)
    fig = plt.figure(figsize=(10, 5))

    plt.plot(range(1, 1+n_epochs), train_losses, label='Train loss')
    plt.plot(range(1, 1+n_epochs), test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if loss_type == 'all':
        plt.title('Train and test global loss')
    elif loss_type == 'loss_point_to_mesh':
        plt.title('Train and test point_to_mesh loss')
    elif loss_type == 'loss_occupancy':
        plt.title('Train and test occupancy loss')
    elif loss_type == 'loss_smoothness':
        plt.title('Train and test smoothness loss')
    elif loss_type == 'loss_curvature':
        plt.title('Train and test curvature loss')

    plt.legend()    
    plt.savefig(f'outputs/figures/{loss_type}_training.png')
    plt.close()