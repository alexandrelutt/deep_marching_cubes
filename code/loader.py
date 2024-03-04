import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def perturb(clean_points, noise):
    num_cells = 32

    perturbed_points = clean_points
    perturbed_points = perturbed_points + np.random.normal(0, noise*num_cells/2, perturbed_points.shape)

    return clean_points, perturbed_points

def load_data(set, noise):
    if set == 'train':
        data = np.load('all_data/points_shapenet_32x32x32_train.npy')
        pts, pts_gt = perturb(data, noise=noise)

        pts = torch.tensor(pts, dtype=torch.float32)
        pts_gt = torch.tensor(pts_gt, dtype=torch.float32)

    else:
        raise NotImplementedError(f'Set {set} not implemented.')
    
    return pts, pts_gt

class CustomDataset(Dataset):
    def __init__(self, clean_points, perturbed_points):
        self.clean_points = clean_points
        self.perturbed_points = perturbed_points

    def __len__(self):
        return len(self.clean_points)

    def __getitem__(self, index):
        return self.clean_points[index], self.perturbed_points[index]

def get_loader(set='train', batch_size=8, noise=0.15):
    clean_points, perturbed_points = load_data(set=set, noise=noise)
    train_set = CustomDataset(clean_points, perturbed_points)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_loader