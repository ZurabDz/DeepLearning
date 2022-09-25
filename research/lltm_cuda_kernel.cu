#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
    new_cell[index] =
        old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}

template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] =
        old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}