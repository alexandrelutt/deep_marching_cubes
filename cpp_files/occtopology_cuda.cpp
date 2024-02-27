#include <torch/extension.h>

#include <vector>

void occ_to_topo_cuda_forward(
    torch::Tensor occupancy,
    torch::Tensor topology);

void occ_to_topo_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor occupancy,
    torch::Tensor topology,
    torch::Tensor grad_occupancy);

void occ_to_topo_forward(
    torch::Tensor occupancy,
    torch::Tensor topology){
    occ_to_topo_cuda_forward(occupancy, topology);
    }

void occ_to_topo_backward(
    torch::Tensor grad_output,
    torch::Tensor occupancy,
    torch::Tensor topology,
    torch::Tensor grad_occupancy){
    occ_to_topo_cuda_backward(grad_output, occupancy, topology, grad_occupancy);
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("occ_to_topo_cuda_forward", &occ_to_topo_forward, "Occupancy To Topology forward (CUDA)");
  m.def("occ_to_topo_cuda_backward", &occ_to_topo_backward, "Occupancy To Topology backward (CUDA)");
}