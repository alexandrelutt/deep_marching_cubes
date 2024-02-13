#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

// Function to get the indices of points in the given cell
torch::Tensor get_points_in_cell(int i, int j, int k, const torch::Tensor& point_cloud, float cell_size) {
    // Calculate the boundaries of the cell
    float min_x = i * cell_size;
    float max_x = (i + 1) * cell_size;
    float min_y = j * cell_size;
    float max_y = (j + 1) * cell_size;
    float min_z = k * cell_size;
    float max_z = (k + 1) * cell_size;
    
    // Create a mask tensor to mark points within the cell
    torch::Tensor mask = (point_cloud.select(1, 0) >= min_x) &
                         (point_cloud.select(1, 0) < max_x) &
                         (point_cloud.select(1, 1) >= min_y) &
                         (point_cloud.select(1, 1) < max_y) &
                         (point_cloud.select(1, 2) >= min_z) &
                         (point_cloud.select(1, 2) < max_z);

    // Find indices where the mask is true
    torch::Tensor indices = torch::nonzero(mask).select(1, 0);
    
    return indices;
}

int grid_pooling(torch::Tensor volumetric_representation, torch::Tensor global_indices, torch::Tensor point_cloud, torch::Tensor point_features_map, int N, int feature_size) {
    /*
    Perform grid pooling operation on the input point features map.

    Args:
    - point_features_map: Input point features map of shape (K, 16), where
                          K is the number of points and 16 is the dimensionality of each point feature.
    - N: Desired size of the output volumetric representation.

    Returns:
    - volumetric_representation: Volumetric representation after grid pooling,
                                 of shape (N, N, N, 16).
    */

    // Calculate the size of each cell
    double cell_size = 1.0 / N;
    int C = point_features_map.size(1);

    // Iterate over each point in the batch
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j){
            for (int k=0; k<N; ++k){
                torch::Tensor local_indices = get_points_in_cell(i, j, k, point_cloud, cell_size);
                // if the grid is not empty, assign max features to the grid and keep index of maximum feature
                if (local_indices.size(0) > 0){
                    auto feat_grid = point_features_map.index_select(0, local_indices);
                    // print size of feat_grid
                    // Max pooling over all points
                    auto max_result = feat_grid.max(0);
                    auto feat_max = std::get<0>(max_result);
                    auto indices_max = std::get<1>(max_result);
                    
                    // store the maximum features in the grid
                    volumetric_representation[i][j][k] = feat_max;

                    // store indices of maximum feature
                    int global_pos = i*(N*N) + j*N + k;
                
                    for (int c=0; c<C; ++c){
                        auto c_index = indices_max.index({c}).item<int>();
                        auto global_c_index = local_indices.index({c_index}).item<int>();
                        global_indices.index_put_({global_pos, c}, global_c_index);

                    }
                }
            }
        }
    }

    return 1;
}

int grid_pooling_backward(torch::Tensor grad_output, torch::Tensor grad_points, torch::Tensor indices){
    int N = grad_output.size(0);
    int C = grad_output.size(1);

    for (int i=0; i<N; i++){
        for (int k=0; k<C; k++){
            int local_index = indices.index({i, k}).item<int>();
            if (local_index != -1){
                float grad = grad_output.index({i, k}).item<float>();
                grad_points.index_put_({local_index, k}, grad);
            }
        }
    }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_pooling", &grid_pooling, "Grid Pooling");
  m.def("grid_pooling_backward", &grid_pooling_backward, "Grid Pooling Backward");
}