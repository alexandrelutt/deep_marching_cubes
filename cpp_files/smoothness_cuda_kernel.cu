#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <THC.h>
#include <THCGeneral.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * calculate the loss between two neigboring occupancy status 
 */	
__global__ void occupancy_connectivity_kernel( const float *occupancy, float *loss ){

  int i=blockIdx.x;
  int j=blockIdx.y;
  int k=blockIdx.z;

  int W=gridDim.x-1;
  int H=gridDim.y-1;
  int D=gridDim.z-1;

  float loss_ = 0.0;

  float p1 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ]; 
 
  if (j<H){
      float p2 = occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ]; 
      // l1 loss
      loss_ += abs(p1-p2);
  }
  if (i<W){
      float p3 = occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ]; 
      // l1 loss
      loss_ += abs(p1-p3);
  }
  if (k<D){
      float p4 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ]; 
      // l1 loss
      loss_ += abs(p1-p4);
  }
  loss[ i*(H+1)*(D+1) + j*(D+1) + k ] = loss_;
}

/**
 * propagate the gradient to the occupancy status 
 */	
__global__ void grad_occupancy_connectivity_kernel( const float *occupancy, float *grad_occupancy ){

  int i=blockIdx.x;
  int j=blockIdx.y;
  int k=blockIdx.z;

  int W=gridDim.x-1;
  int H=gridDim.y-1;
  int D=gridDim.z-1;

  float p1 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ]; 
 
  if (j<H){
      float p2 = occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ]; 
      // l1 loss
      float sign;
      if (p1>p2){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ], -sign );

  }
  if (i<W){
      float p3 = occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ]; 
      // l1 loss
      float sign;
      if (p1>p3){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ], -sign );
  }
  if (k<D){
      float p4 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ]; 
      float sign;
      if (p1>p4){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ], -sign );
  }
}

void connectivity_cuda_forward(
    torch::Tensor occupancy,
    torch::Tensor loss){

    int N = occupancy.size(0);

    dim3 dimGrid(N, N, N);

    // THCudaTensor *loss_all = THCudaTensor_newWithSize1d(state, W*H*D);
    // THCudaTensor_zero(state, loss_all);
    // // lauch the kernel
    // occupancy_connectivity_kernel<<< dimGrid, 1, 0, THCState_getCurrentStream(state) >>>(
    //         THCudaTensor_data(state, occupancy),
    //         THCudaTensor_data(state, loss_all) );

    // float loss_= THCudaTensor_sumall(state, loss_all);

    // THCudaTensor_set1d(state, loss, 0, loss_);

}


void connectivity_cuda_backward(
    torch::Tensor grad_loss,
    torch::Tensor grad_occupancy,
    torch::Tensor occupancy){

    int N = occupancy.size(0);

    dim3 dimGrid(N, N, N);

    // grad_occupancy_connectivity_kernel<<< dimGrid, 1, 0, THCState_getCurrentStream(state) >>>(
    //         THCudaTensor_data(state, occupancy),
    //         THCudaTensor_data(state, grad_occupancy) );

    // float grad_output_=THCudaTensor_get1d(state, grad_output, 0);
    // THCudaTensor_mul(state, grad_occupancy, grad_occupancy, grad_output_);

}