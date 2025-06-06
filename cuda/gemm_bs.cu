#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define CHECK_CUDA(call)                                            \
  {                                                                 \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess)                                         \
    {                                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  }

// WMMA tile sizes
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Kernel for batched matrix multiplication using WMMA Tensor Cores
__global__ void wmmaBatchedGemmKernel(half *a, half *b, float *c,
                                      int M, int N, int K, int batch_size)
{
  int batch_id = blockIdx.z; // batch index

  // Leading dimensions for each matrix
  int lda = K;
  int ldb = N;
  int ldc = N;

  // Calculate warp tile indices (each warp computes one WMMA tile)
  int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
  int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

  int row = warpM * WMMA_M;
  int col = warpN * WMMA_N;

  // Pointers offset by batch
  half *a_batch = a + batch_id * M * K;
  half *b_batch = b + batch_id * K * N;
  float *c_batch = c + batch_id * M * N;

  // Declare fragments
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  fill_fragment(acc_frag, 0.0f);

  for (int k = 0; k < K; k += WMMA_K)
  {
    // Load A tile with boundary checks and zero padding
    half a_tile[WMMA_M * WMMA_K];
#pragma unroll
    for (int i = 0; i < WMMA_M; i++)
    {
#pragma unroll
      for (int j = 0; j < WMMA_K; j++)
      {
        int r = row + i;
        int cA = k + j;
        if (r < M && cA < K)
          a_tile[i * WMMA_K + j] = a_batch[r * lda + cA];
        else
          a_tile[i * WMMA_K + j] = __float2half(0.0f);
      }
    }
    load_matrix_sync(a_frag, a_tile, WMMA_K);

    // Load B tile with boundary checks and zero padding
    half b_tile[WMMA_K * WMMA_N];
#pragma unroll
    for (int i = 0; i < WMMA_K; i++)
    {
#pragma unroll
      for (int j = 0; j < WMMA_N; j++)
      {
        int rB = k + i;
        int c = col + j;
        if (rB < K && c < N)
          b_tile[i * WMMA_N + j] = b_batch[rB * ldb + c];
        else
          b_tile[i * WMMA_N + j] = __float2half(0.0f);
      }
    }
    load_matrix_sync(b_frag, b_tile, WMMA_N);

    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Store results with boundary checks
  if (row < M && col < N)
  {
    float c_tile[WMMA_M * WMMA_N];
#pragma unroll
    for (int i = 0; i < WMMA_M; i++)
    {
#pragma unroll
      for (int j = 0; j < WMMA_N; j++)
      {
        int r = row + i;
        int c_ = col + j;
        if (r < M && c_ < N)
          c_tile[i * WMMA_N + j] = c_batch[r * ldc + c_];
        else
          c_tile[i * WMMA_N + j] = 0.0f;
      }
    }

#pragma unroll
    for (int i = 0; i < acc_frag.num_elements; i++)
    {
      c_tile[i] += acc_frag.x[i];
    }

#pragma unroll
    for (int i = 0; i < WMMA_M; i++)
    {
#pragma unroll
      for (int j = 0; j < WMMA_N; j++)
      {
        int r = row + i;
        int c_ = col + j;
        if (r < M && c_ < N)
          c_batch[r * ldc + c_] = c_tile[i * WMMA_N + j];
      }
    }
  }
}

// Host helper functions remain the same, with batch dimension added

void init_host_matrices(half *a, half *b, int M, int N, int K, int batch_size)
{
  for (int batch = 0; batch < batch_size; batch++)
  {
    for (int i = 0; i < M * K; i++)
    {
      float val = static_cast<float>(rand()) / RAND_MAX;
      a[batch * M * K + i] = __float2half(val);
    }
    for (int i = 0; i < K * N; i++)
    {
      float val = static_cast<float>(rand()) / RAND_MAX;
      b[batch * K * N + i] = __float2half(val);
    }
  }
}

void cpu_gemm(const half *a, const half *b, float *c,
              int M, int N, int K, int batch_size)
{
  for (int batch = 0; batch < batch_size; batch++)
  {
    const half *a_batch = a + batch * M * K;
    const half *b_batch = b + batch * K * N;
    float *c_batch = c + batch * M * N;
    for (int i = 0; i < M; i++)
    {
      for (int j = 0; j < N; j++)
      {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
          sum += __half2float(a_batch[i * K + k]) * __half2float(b_batch[k * N + j]);
        }
        c_batch[i * N + j] = sum;
      }
    }
  }
}

bool verify_results(float *host_c, float *gpu_c, int M, int N, int batch_size)
{
  const float epsilon = 1e-2f;
  for (int batch = 0; batch < batch_size; batch++)
  {
    for (int i = 0; i < M * N; i++)
    {
      float diff = abs(host_c[batch * M * N + i] - gpu_c[batch * M * N + i]);
      if (diff > epsilon)
      {
        printf("Mismatch at batch %d index %d: CPU=%f, GPU=%f\n",
               batch, i, host_c[batch * M * N + i], gpu_c[batch * M * N + i]);
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char *argv[])
{
  // Matrix sizes and batch size
  int batch_size = 4;
  int M = 128;
  int N = 256;
  int K = 512;

  printf("Batch size: %d, Matrix sizes: M=%d, N=%d, K=%d\n", batch_size, M, N, K);

  // Host allocations
  half *host_a = (half *)malloc(sizeof(half) * batch_size * M * K);
  half *host_b = (half *)malloc(sizeof(half) * batch_size * K * N);
  float *host_c = (float *)malloc(sizeof(float) * batch_size * M * N);
  float *host_c_ref = (float *)malloc(sizeof(float) * batch_size * M * N);

  // Initialize inputs
  srand(0);
  init_host_matrices(host_a, host_b, M, N, K, batch_size);

  // Initialize output
  for (int i = 0; i < batch_size * M * N; i++)
  {
    host_c[i] = 0.0f;
    host_c_ref[i] = 0.0f;
  }

  // Device allocations
  half *dev_a, *dev_b;
  float *dev_c;
  CHECK_CUDA(cudaMalloc(&dev_a, sizeof(half) * batch_size * M * K));
  CHECK_CUDA(cudaMalloc(&dev_b, sizeof(half) * batch_size * K * N));
  CHECK_CUDA(cudaMalloc(&dev_c, sizeof(float) * batch_size * M * N));

  // Copy inputs to device
  CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(half) * batch_size * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(half) * batch_size * K * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_c, host_c, sizeof(float) * batch_size * M * N, cudaMemcpyHostToDevice));

  // Kernel launch configuration
  dim3 threadsPerBlock(32, 4); // 4 warps per block
  dim3 numBlocks(
      (N + WMMA_N * threadsPerBlock.x / 32 - 1) / (WMMA_N * threadsPerBlock.x / 32),
      (M + WMMA_M * threadsPerBlock.y - 1) / (WMMA_M * threadsPerBlock.y),
      batch_size); // Use z-dim for batch

  printf("Launching kernel with grid (%d,%d,%d), block (%d,%d)\n",
         numBlocks.x, numBlocks.y, numBlocks.z, threadsPerBlock.x, threadsPerBlock.y);

  wmmaBatchedGemmKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, M, N, K, batch_size);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy results back
  CHECK_CUDA(cudaMemcpy(host_c, dev_c, sizeof(float) * batch_size * M * N, cudaMemcpyDeviceToHost));

  // CPU reference
  printf("Computing reference on CPU...\n");
  cpu_gemm(host_a, host_b, host_c_ref, M, N, K, batch_size);

  // Verify
  bool ok = verify_results(host_c_ref, host_c, M, N, batch_size);
  printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

  // Cleanup
  free(host_a);
  free(host_b);
  free(host_c);
  free(host_c_ref);
  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));

  return ok ? 0 : 1;
}
