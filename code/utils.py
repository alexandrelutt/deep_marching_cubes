import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, loss_type='all'):
    n_epochs = len(train_losses)
    fig = plt.figure(figsize=(10, 5))

    plt.plot(range(n_epochs), train_losses, label='Train loss')
    plt.plot(range(n_epochs), test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if loss_type == 'all':
        plt.title('Train and test global loss')
    elif loss_type == 'point_to_mesh':
        plt.title('Train and test point_to_mesh loss')
    elif loss_type == 'occupancy':
        plt.title('Train and test occupancy loss')
    elif loss_type == 'smoothness':
        plt.title('Train and test smoothness loss')
    elif loss_type == 'curvature':
        plt.title('Train and test curvature loss')

    plt.legend()

    plt.show()