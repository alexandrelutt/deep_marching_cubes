import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
from sklearn.neighbors import KDTree

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

def save_ply(pts, filename):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(filename, pcd)

def save_occupancy_fig(input_pts, pts, occupancy, grid, i):
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

    ax.scatter(input_pts[:, 0], input_pts[:, 1], input_pts[:, 2], '.',
            color='#727272', zorder=1)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_input_occupancy_{i}.png')
    save_ply(input_pts, filename=f'outputs/points/test_example_input_occupancy_{i}.ply')

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], '.',
            color='#727272', zorder=1)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_true_occupancy_{i}.png')
    save_ply(pts, filename=f'outputs/points/test_example_true_occupancy_{i}.ply')

    fig = plt.figure()
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')
    rgba_x = np.zeros((len(xv_cls), 4))
    rgba_x[:, 0] = 1.0
    rgba_x[:, 3] = occupancy.flatten()

    proba_treshold = 0.4

    pred_points = np.array([xv_cls, yv_cls, zv_cls]).T
    pred_points = pred_points[occupancy.flatten() > proba_treshold]

    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], '.', color=rgba_x[occupancy.flatten() > proba_treshold], zorder=1)

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_pred_occupancy_{i}.png')
    save_ply(pred_points, filename=f'outputs/points/test_example_pred_occupancy_{i}.ply')

def save_mesh_fig(pts, offset, topology, grid, i):
    num_cells = len(grid)-1
    _, topology_max = torch.max(topology, dim=1)

    topology_max = topology_max.view(num_cells, num_cells, num_cells)

    vertices = torch.zeros(num_cells**3 * 12, 3).float()
    faces = torch.zeros(num_cells**3 * 12, 3).float()

    vertices, faces, num_vertices, num_faces = cpp_utils.pred_to_mesh(offset.detach().cpu(), topology_max.detach().cpu(), vertices, faces)

    vertices = vertices[:num_vertices, :].numpy()
    faces = faces[:num_faces, :].numpy()

    vertices = np.asarray(vertices)
    vertices_unique, indices = unique_rows(vertices)
    faces = np.asarray(faces).flatten()
    faces_unique = faces[indices].reshape((-1, 3))

    vertices_unique = vertices_unique[:, [2, 0, 1]]

    save_mesh_helper(vertices_unique, faces_unique, i)

    fig = plt.figure(0)
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    color = [0.8, 0.5, 0.5]
    ax.plot_trisurf(vertices_unique[:, 0],
                    vertices_unique[:, 1],
                    vertices_unique[:, 2],
                    triangles=faces_unique,
                    color=color,
                    edgecolor='none',
                    alpha=1.0
                    )

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(grid.min(), grid.max())
    ax.set_zlim(grid.min(), grid.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.savefig(f'outputs/figures/test_example_mesh_{i}.png')

def get_chamfer_dist(true_points, occupancy, grid):
    xv_cls, yv_cls, zv_cls = np.meshgrid(
            range(len(grid)),
            range(len(grid)),
            range(len(grid)),
            indexing='ij')
    xv_cls = xv_cls.flatten()
    yv_cls = yv_cls.flatten()
    zv_cls = zv_cls.flatten()

    proba_treshold = 0.4
    pred_points = np.array([xv_cls, yv_cls, zv_cls]).T
    pred_points = pred_points[occupancy.flatten() > proba_treshold]

    array1 = true_points.cpu().numpy()
    array2 = pred_points.cpu().numpy()

    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

def get_hamming_dist(clean_batch, occupancy, device):
    return 0

def visualize(model, test_loader, device):
    model.to(device)
    avg_chamfer_dist = 0
    avg_hamming_dist = 0
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
                    perturbed_batch[-1].data.cpu().numpy(),
                    clean_batch[-1].data.cpu().numpy(),
                    occupancy[-1].data.cpu().numpy(),
                    np.arange(0, 32+1),
                    i)

            topology_vis = topology[:, :, torch.LongTensor(get_accepted_topologies())]

            save_mesh_fig(
                    clean_batch[-1].data.cpu().numpy(),
                    offset[-1],
                    topology_vis[-1],
                    np.arange(0, 32+1),
                    i)
            
            avg_chamfer_dist += get_chamfer_dist(clean_batch.data.cpu().numpy(),
                                                 occupancy.data.cpu().numpy(), 
                                                 np.arange(0, 32+1)
                                                )
            avg_hamming_dist += get_hamming_dist(clean_batch, occupancy)
        
    avg_chamfer_dist /= len(test_loader)
    avg_hamming_dist /= len(test_loader)

    return avg_chamfer_dist, avg_hamming_dist