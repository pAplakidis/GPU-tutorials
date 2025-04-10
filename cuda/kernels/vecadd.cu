extern "C" __global__ void vec_add_kernel(float *A, float *B, float *C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}