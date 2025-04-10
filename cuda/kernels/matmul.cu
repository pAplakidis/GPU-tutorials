extern "C" __global__ void matrix_mul(float *a, float *b, float *c, int N)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  __shared__ float a_shared[% d][% d];
  __shared__ float b_shared[% d][% d];

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;
  float sum = 0.0f;

  for (int m = 0; m < N / % d; ++m)
  {
    a_shared[ty][tx] = a[row * N + (m * % d + tx)];
    b_shared[ty][tx] = b[(m * % d + ty) * N + col];
    __syncthreads();

    for (int k = 0; k < % d; ++k)
    {
      sum += a_shared[ty][k] * b_shared[k][tx];
    }
    __syncthreads();
  }

  c[row * N + col] = sum;
}