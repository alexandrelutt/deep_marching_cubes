#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

torch::Tensor grid_pooling(torch::Tensor point_cloud, torch::Tensor point_features_map, int N) {
    /*
    Perform grid pooling operation on the input point features map.

    Args:
    - point_features_map: Input point features map of shape (B, K, 16), where
                          B is the batch size, K is the number of points,
                          and 16 is the dimensionality of each point feature.
    - N: Desired size of the output volumetric representation.

    Returns:
    - volumetric_representation: Volumetric representation after grid pooling,
                                 of shape (B, N, N, N, 16).
    */

    int B = point_features_map.size(0);
    int K = point_features_map.size(1);
    int D = 16;  // Dimensionality of each point feature

    // Initialize the volumetric representation with zeros
    auto volumetric_representation = torch::zeros({B, N, N, N, D});

    // Calculate the size of each cell
    double cell_size = 1.0 / N;

    // Iterate over each batch
    for (int b = 0; b < B; ++b) {
        // Iterate over each point in the batch
        for (int k = 0; k < K; ++k) {
            // Calculate the cell indices for the current point
            auto point = point_cloud[b][k];
            int x = static_cast<int>(std::floor(point[0].item<double>() / cell_size));
            int y = static_cast<int>(std::floor(point[1].item<double>() / cell_size));
            int z = static_cast<int>(std::floor(point[2].item<double>() / cell_size));

            // Perform max pooling within the cell
            volumetric_representation[b][x][y][z] = torch::max(
                volumetric_representation[b][x][y][z],
                point_features_map[b][k]
            );
        }
    }

    return volumetric_representation;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_pooling", &grid_pooling, "Grid Pooling");
}