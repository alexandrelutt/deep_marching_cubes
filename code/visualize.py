import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from code.table import get_accepted_topologies
from utils_extension import cpp_utils

def save_mesh_helper(vertices, faces, i):
    filename = f'outputs/meshes/test_example_mesh_{i}.off'
    with open(filename, 'w') as f:
        f.write('OFF\n')

        n_vertice = vertices.shape[0] 
        n_face = faces.shape[0] 
        f.write('%d %d 0\n' % (n_vertice, n_face))
        for nv in range(n_vertice):
            f.write('%f %f %f\n' % (vertices[nv,1], vertices[nv,2], vertices[nv,0]))
        for nf in range(n_face):
            f.write('3 %d %d %d\n' % (faces[nf,0], faces[nf,1], faces[nf,2]))

def unique_rows(a):
    rowtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    b = np.ascontiguousarray(a).view(rowtype)
    _, idx, inverse = np.unique(b, return_index=True, return_inverse=True)
    return a[idx], inverse 

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

    plt.savefig(f'outputs/figures/test_example_occupancy_{i}.png')

def save_mesh_fig(pts, offset, topology, grid, i, save_mesh=True):
    num_cells = len(grid)-1
    _, topology_max = torch.max(topology, dim=1)
    topology_max = topology_max.view(num_cells, num_cells, num_cells)

    vertices = torch.FloatTensor(num_cells**3 * 12, 3)
    faces = torch.FloatTensor(num_cells**3 * 12, 3)
    num_vertices = torch.LongTensor(1)
    num_faces = torch.LongTensor(1)

    print(cpp_utils.pred_to_mesh(offset.data.cpu(), topology_max.data.cpu(),
            vertices, faces, num_vertices, num_faces))

    vertices = vertices[0:num_vertices[0], :].numpy()
    faces = faces[0:num_faces[0], :].numpy()

    vertices = np.asarray(vertices)
    vertices_unique, indices = unique_rows(vertices)
    faces = np.asarray(faces).flatten()
    faces_unique = faces[indices].reshape((-1, 3))

    if save_mesh:
        save_mesh_helper(vertices_unique, faces_unique, i)

    xv_cls, yv_cls, zv_cls = np.meshgrid(grid[:-1], grid[:-1], grid[:-1], indexing='ij')
    xv_cls = xv_cls.flatten()
    yv_cls = yv_cls.flatten()
    zv_cls = zv_cls.flatten()

    fig = plt.figure(0)
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    # plot the scattered points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], '.', color='#727272', zorder=1)

    # plot the mesh
    color = [0.8, 0.5, 0.5]
    ax.plot_trisurf(vertices_unique[:, 0],
                        vertices_unique[:, 1],
                        vertices_unique[:, 2],
                        triangles=faces_unique,
                        color=color,
                        edgecolor='none',
                        alpha=1.0)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_mesh_{i}.png')


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
            # save_occupancy_fig(
            #         clean_batch[-1].data.cpu().numpy(),
            #         occupancy[-1].data.cpu().numpy(),
            #         np.arange(0, 32+1),
            #         i)

            topology_vis = topology[:, :, torch.LongTensor(get_accepted_topologies())]

            save_mesh_fig(
                    clean_batch[-1].data.cpu().numpy(),
                    offset[-1],
                    topology_vis[-1],
                    np.arange(0, 32+1),
                    i)

            if i > 3:
                break