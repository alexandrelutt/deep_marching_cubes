import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from code.table import get_accepted_topologies

def save_occupancy_fig(pts, occupancy, grid, i):
    xv_cls, yv_cls, zv_cls = np.meshgrid(
            range(len(grid)),
            range(len(grid)),
            range(len(grid)),
            indexing='ij')
    xv_cls = xv_cls.flatten()
    yv_cls = yv_cls.flatten()
    zv_cls = zv_cls.flatten()

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], '.',
            color='#727272', zorder=1)
    
    rgba_x = np.zeros((len(xv_cls), 4))
    rgba_x[:, 0] = 1.0
    rgba_x[:, 3] = occupancy.flatten()
    
    ax.scatter(xv_cls, yv_cls, zv_cls, '.', color=rgba_x, zorder=1)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_{i}.png')

# def save_mesh_fig(pts, offset, topology, grid, i):
#     num_cells = len(grid)-1
#     _, topology_max = torch.max(topology, dim=1)
#     topology_max = topology_max.view(num_cells, num_cells, num_cells)

#     vertices = torch.FloatTensor(num_cells**3 * 12, 3)
#     faces = torch.FloatTensor(num_cells**3 * 12, 3)
#     num_vertices = torch.LongTensor(1)
#     num_faces = torch.LongTensor(1)

#     pred_to_mesh(offset.data.cpu(), topology_max.data.cpu(),
#             vertices, faces, num_vertices, num_faces)



def visualize(model, test_loader, device):
    model.to(device)
    with torch.no_grad():
        for i, (clean_batch, perturbed_batch) in tqdm(enumerate(test_loader)):
            clean_batch = clean_batch.to(device)
            perturbed_batch = perturbed_batch.to(device)

            offset, topology, occupancy = model(perturbed_batch)

            topology_fused = topology[-1].data.cpu().numpy()
            topology_fused = np.maximum(topology_fused[:, 0:128],
                                        topology_fused[:, 256:127:-1])
            topology_fused = topology_fused[:, get_accepted_topologies()]
            save_occupancy_fig(
                    clean_batch[-1].data.cpu().numpy(),
                    occupancy[-1].data.cpu().numpy(),
                    np.arange(0, 32+1),
                    i)

            topology_vis = topology[:, :, torch.LongTensor(get_accepted_topologies())]

            # save_mesh_fig(
            #         clean_batch[-1].data.cpu().numpy(),
            #         offset[-1],
            #         topology_vis[-1],
            #         np.arange(0, 32+1),
            #         i)

            if i > 3:
                break