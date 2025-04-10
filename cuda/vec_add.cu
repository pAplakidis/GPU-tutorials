#include <stdio.h>

#define CUDA_CHECK(err)                                                              \
  {                                                                                  \
    if (err != cudaSuccess)                                                          \
    {                                                                                \
      printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                            \
    }                                                                                \
  }

// tutorial from [ https://0mean1sigma.com/what-is-gpgpu-programming/ ]

/*
The blocks in a grid and the threads in each block can be organized as 1D (x), 2D (x,y), or 3D (x,y,z).
The data structure dictates whether 1D would be sufficient or 2D/3D organization is required.
In this example, the data is a 1D vector, so a 1D grid and 1D blocks should work well.
I decided to go with 4 threads in each block.
As there are n=10 total elements, it means that I will need a total of 3 blocks.
Remember that each block must have the same number of threads, so in doing this, the program will spawn a total of 12 threads.
All the blocks are indexed, i.e., in this case, I will have block indices ranging from 0 to 2.
The threads in each block are indexed as well.
The thing to note is that thread indices are local to each block, i.e., each block will have thread indices ranging from 0 to 3.

The custom-defined CUDA type dim3 defines

The number of threads in each block: dim3 dim_block(4, 1, 1).
Note that in this case, the y and z dimensions are set to 1.
The number of blocks in the grid can then be decided based on the length of vectors and the total threads in each block: dim3 dim_grid(ceil(N/4.0), 1, 1)
*/

__global__ void vec_add_kernel(float *A, float *B, float *C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N)
  {
    C[i] = A[i] + B[i];
  }
}

int main()
{
  int N = 10;

  float *A = (float *)malloc(N * sizeof(float));
  float *B = (float *)malloc(N * sizeof(float));
  float *C = (float *)malloc(N * sizeof(float));

  if (!(A && B && C))
  {
    perror("Error allocating memory for vectors\n");
    return 1;
  }

  for (int i = 0; i < N; i++)
  {
    A[i] = (float)(rand() % (10 - 0 + 1) + 0);
    B[i] = (float)(rand() % (10 - 0 + 1) + 0);
    C[i] = 0;
  }

  // allocate device memory
  float *d_A;
  cudaError_t err_A = cudaMalloc((void **)&d_A, N * sizeof(float));
  CUDA_CHECK(err_A);

  float *d_B;
  cudaError_t err_B = cudaMalloc((void **)&d_B, N * sizeof(float));
  CUDA_CHECK(err_B);

  float *d_C;
  cudaError_t err_C = cudaMalloc((void **)&d_C, N * sizeof(float));
  CUDA_CHECK(err_C);

  // RAM => VRAM
  cudaError_t err_A_ = cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK(err_A_);
  cudaError_t err_B_ = cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK(err_B_);

  // execute kernel
  dim3 dim_block(4, 1, 1); // define number of threads in a block
  // dim3 dim_grid(ceil((N + 3) / 4), 1, 1); // define number of blocks in a grid
  dim3 dim_grid((N + 3) / 4, 1, 1);
  vec_add_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);

  cudaError_t err_kernel = cudaGetLastError(); // Capture kernel launch error
  CUDA_CHECK(err_kernel);                      // This will print the error if there's an issue
  cudaDeviceSynchronize();                     // Ensure execution completes

  cudaError_t err_C_ = cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
  CUDA_CHECK(err_C_);

  for (int i = 0; i < N; i++)
  {
    printf("%f\t", C[i]);
  }
  printf("\n");

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);

  free(C);
  free(B);
  free(A);

  return 0;
}
