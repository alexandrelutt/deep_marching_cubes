#include <torch/extension.h>

#include <vector>

void connectivity_cuda_forward(
    torch::Tensor occupancy
    torch::Tensor loss);


void connectivity_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor occupancy,
    torch::Tensor grad_occupancy);

void connectivity_forward(
    torch::Tensor occupancy,
    torch::Tensor loss){
    connectivity_cuda_forward(occupancy);
}

void connectivity_backward(
    torch::Tensor grad_output,
    torch::Tensor occupancy,
    torch::Tensor grad_occupancy){
    connectivity_cuda_backward(grad_output, occupancy, grad_occupancy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("connectivity_cuda_forward", &connectivity_forward, "Smoothness forward (CUDA)");
  m.def("connectivity_cuda_backward", &connectivity_backward, "Smoothness backward (CUDA)");
}