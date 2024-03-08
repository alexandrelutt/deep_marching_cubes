#include <torch/extension.h>

#include <vector>

torch::Tensor curvature_cuda_forward(
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable,
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable);

void curvature_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable,
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable,
    torch::Tensor grad_offset);

torch::Tensor curvature_forward(
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable,
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable){
    return curvature_cuda_forward(offset, topology, xTable, yTable, zTable, innerTable);
    }

void curvature_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable,
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable,
    torch::Tensor grad_offset){
    curvature_cuda_backward(grad_output, offset, topology, xTable, yTable, zTable, innerTable, grad_offset);
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("curvature_constraint_cuda_forward", &curvature_forward, "Curvature forward (CUDA)");
  m.def("curvature_constraint_cuda_backward", &curvature_backward, "Curvature backward (CUDA)");
}