#include <torch/extension.h>

#include <vector>

void dist_cuda_forward(
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor distances_full,
    torch::Tensor indices);

void dist_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor indices,
    torch::Tensor grad_offset);

void dist_forward(
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor distances_full,
    torch::Tensor indices){
    dist_cuda_forward(offset, points, distances_full, indices);
    }

void dist_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor indices,
    torch::Tensor grad_offset){
    dist_cuda_backward(grad_output, offset, points, indices, grad_offset);
    }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pt_topo_distance_cuda_forward", &dist_forward, "Distance from points to topology (CUDA)");
  m.def("pt_topo_distance_cuda_backward", &dist_backward, "Distance from points to topology backward (CUDA)");
}