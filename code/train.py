import numpy as np
import torch
import time
from tqdm import tqdm
from google.cloud import storage

from code.utils import plot_losses

def train(model, train_loader, test_loader, loss_module, n_epochs, optimizer, scheduler, device):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket('npm3d')

    model.to(device)
    train_losses, test_losses = [], []
    train_loss_point_to_mesh, train_loss_occupancy, train_loss_smoothness, train_loss_curvature = [], [], [], []
    test_loss_point_to_mesh, test_loss_occupancy, test_loss_smoothness, test_loss_curvature = [], [], [], []
    best_test_loss = np.inf

    for t in range(n_epochs):
        t0 = time.time()
        print(f'-------------------------------\nEpoch {t+1}/{n_epochs}\n-------------------------------')
        epoch_train_loss = 0
        epoch_train_loss_point_to_mesh = 0
        epoch_train_loss_occupancy = 0
        epoch_train_loss_smoothness = 0
        epoch_train_loss_curvature = 0

        for (clean_batch, perturbed_batch) in tqdm(train_loader):
            clean_batch = clean_batch.to(device)
            perturbed_batch = perturbed_batch.to(device)

            offset, topology, occupancy = model(perturbed_batch)
            loss, loss_point_to_mesh, loss_occupancy, loss_smoothness, loss_curvature = loss_module.loss(offset, topology, clean_batch, occupancy)

            print(f'loss: {loss}')
            print(f'loss_point_to_mesh: {loss_point_to_mesh}')
            print(f'loss_occupancy: {loss_occupancy}')
            print(f'loss_smoothness: {loss_smoothness}')

            epoch_train_loss += loss.item()
            epoch_train_loss_point_to_mesh += loss_point_to_mesh.item()
            epoch_train_loss_occupancy += loss_occupancy.item()
            epoch_train_loss_smoothness += loss_smoothness.item()
            epoch_train_loss_curvature += loss_curvature.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_train_loss /= len(train_loader)
        epoch_train_loss_point_to_mesh /= len(train_loader)
        epoch_train_loss_occupancy /= len(train_loader)
        epoch_train_loss_smoothness /= len(train_loader)
        epoch_train_loss_curvature /= len(train_loader)

        scheduler.step(epoch_train_loss)

        train_losses.append(epoch_train_loss)
        train_loss_point_to_mesh.append(epoch_train_loss_point_to_mesh)
        train_loss_occupancy.append(epoch_train_loss_occupancy)
        train_loss_smoothness.append(epoch_train_loss_smoothness)
        train_loss_curvature.append(epoch_train_loss_curvature)

        with torch.no_grad():
            epoch_test_loss = 0
            epoch_test_loss_point_to_mesh = 0
            epoch_test_loss_occupancy = 0
            epoch_test_loss_smoothness = 0
            epoch_test_loss_curvature = 0
            for (clean_batch, perturbed_batch) in test_loader:
                clean_batch = clean_batch.to(device)
                perturbed_batch = perturbed_batch.to(device)

                offset, topology, occupancy = model(perturbed_batch)
                loss, loss_point_to_mesh, loss_occupancy, loss_smoothness, loss_curvature = loss_module.loss(offset, topology, clean_batch, occupancy)

                epoch_test_loss += loss.item()
                epoch_test_loss_point_to_mesh += loss_point_to_mesh.item()
                epoch_test_loss_occupancy += loss_occupancy.item()
                epoch_test_loss_smoothness += loss_smoothness.item()
                epoch_test_loss_curvature += loss_curvature.item()

            epoch_test_loss /= len(test_loader)
            epoch_test_loss_point_to_mesh /= len(test_loader)
            epoch_test_loss_occupancy /= len(test_loader)
            epoch_test_loss_smoothness /= len(test_loader)
            epoch_test_loss_curvature /= len(test_loader)

            test_losses.append(epoch_test_loss)
            test_loss_point_to_mesh.append(epoch_test_loss_point_to_mesh)
            test_loss_occupancy.append(epoch_test_loss_occupancy)
            test_loss_smoothness.append(epoch_test_loss_smoothness)
            test_loss_curvature.append(epoch_test_loss_curvature)
            
        print(f'Training loss: {epoch_train_loss}.')
        print(f'Test loss:     {epoch_test_loss}.')

        if epoch_test_loss < best_test_loss:
            print('\n  New best model has been found!')
            best_test_loss = epoch_test_loss
            best_model = model
            torch.save(model.state_dict(), 'outputs/models/best_model.pth')
            print('  New best model has been saved.\n')

        train_losses_dict = {
            'all': train_losses,
            'loss_point_to_mesh': train_loss_point_to_mesh,
            'loss_occupancy': train_loss_occupancy,
            'loss_smoothness': train_loss_smoothness,
            'loss_curvature': train_loss_curvature
        }
        test_losses_dict = {
            'all': test_losses,
            'loss_point_to_mesh': test_loss_point_to_mesh,
            'loss_occupancy': test_loss_occupancy,
            'loss_smoothness': test_loss_smoothness,
            'loss_curvature': test_loss_curvature
        }

        print('Now plotting losses...')
        for key, value in train_losses_dict.items():
            plot_losses(value, test_losses_dict[key], loss_type=key)
            blob = bucket.blob(f'figures/{key}_training.png')
            blob.upload_from_filename(f'outputs/figures/{key}_training.png')

        print('Done!')

        dt = time.time() - t0
        print(f'\nEpoch duration: {dt:.2f}s.\n')

    print('Training complete!\n')

    return train_losses_dict, test_losses_dict, best_model