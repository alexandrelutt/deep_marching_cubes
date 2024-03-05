import numpy as np
import torch
import time

def train(model, train_loader, test_loader, loss_module, n_epochs, optimizer, device):
    print(f'Starting training for {n_epochs} epochs.\n')
    model.to(device)
    train_losses, test_losses = [], []
    best_test_loss = np.inf

    for t in range(n_epochs):
        t0 = time.time()
        print(f'-------------------------------\nEpoch {t+1}\n-------------------------------')
        epoch_train_loss = 0
        for i, (clean_batch, perturbed_batch) in enumerate(train_loader):
            clean_batch = clean_batch.to(device)
            perturbed_batch = perturbed_batch.to(device)

            offset, topology, occupancy = model(perturbed_batch)

            loss = loss_module.loss(offset, topology, clean_batch, occupancy)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if i > 1:
                break
            
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

            test_losses.append(epoch_test_loss)
            
        print(f'Training loss: {epoch_train_loss}.')
        print(f'Test loss:     {epoch_test_loss}.')
        if epoch_test_loss < best_test_loss:
            print('  New best model has been found!')
            best_test_loss = epoch_test_loss
            best_model = model
            torch.save(model.state_dict(), 'models/best_model.pth')
            print('  New best model has been saved.')

        dt = time.time() - t0
        print(f'Duration: {dt:.2f}s.\n')

    print('Training complete.')
    return train_losses, test_losses, best_model