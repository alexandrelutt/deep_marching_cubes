import numpy as np
import torch
from tqdm import tqdm

def train(model, train_loader, test_loader, loss_module, n_epochs, device):
    print(f'Now training model for {n_epochs} epochs.\n')
    model.to(device)
    train_losses, test_losses = [], []
    best_test_loss = np.inf

    for t in range(n_epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        epoch_train_loss = 0
        for i, (clean_batch, perturbed_batch) in tqdm(enumerate(train_loader)):
            clean_batch = clean_batch.to(device)
            perturbed_batch = perturbed_batch.to(device)

            offset, topology, occupancy = model(perturbed_batch)

            loss = loss_module.loss(offset, topology, clean_batch, occupancy)

            model.zero_grad()
            loss.backward()

            print(f'Batch loss: {loss.item()}')

            epoch_train_loss += loss.item()

            if i > 1:
                break
            

        print(f'Epoch {t+1} loss: {epoch_train_loss}')
        train_losses.append(epoch_train_loss)

        with torch.no_grad():
            epoch_test_loss = 0
            for i, (clean_batch, perturbed_batch) in enumerate(test_loader):
                clean_batch = clean_batch.to(device)
                perturbed_batch = perturbed_batch.to(device)

                offset, topology, occupancy = model(perturbed_batch)

                loss = loss_module.loss(offset, topology, clean_batch, occupancy)

                epoch_test_loss += loss.item()

                if i > 1:
                    break

            print(f'Test loss: {epoch_test_loss}')
            test_losses.append(epoch_test_loss)
            if epoch_test_loss < best_test_loss:
                print('New best model found.')
                best_test_loss = epoch_test_loss
                # torch.save(model.state_dict(), '/models/best_model.pth')
                # print('Model saved.')

    return train_losses, test_losses