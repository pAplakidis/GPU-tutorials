import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray, compiler

MATRIX_SIZE = 1024  # Must be divisible by BLOCK_SIZE
BLOCK_SIZE = 16

# GPU memory allocation (persistent)
a_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
b_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# CUDA kernel with shared memory optimization
kernel_code = """
__global__ void matrix_mul(float *a, float *b, float *c, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    __shared__ float a_shared[%d][%d];
    __shared__ float b_shared[%d][%d];
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float sum = 0.0f;
    
    for (int m = 0; m < N/%d; ++m) {
        a_shared[ty][tx] = a[row * N + (m * %d + tx)];
        b_shared[ty][tx] = b[(m * %d + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < %d; ++k) {
            sum += a_shared[ty][k] * b_shared[k][tx];
        }
        __syncthreads();
    }
    
    c[row * N + col] = sum;
}
""" % (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

# Compile and get kernel function
module = compiler.SourceModule(kernel_code)
matrix_mul = module.get_function("matrix_mul")

# Initialize matrices directly on GPU (persistent)
a_gpu.set(np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32))
b_gpu.set(np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32))

# Configure grid/block dimensions
grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE)
block = (BLOCK_SIZE, BLOCK_SIZE, 1)

# Execute kernel with timing
start = drv.Event()
end = drv.Event()
start.record()

matrix_mul(a_gpu, b_gpu, c_gpu, np.int32(MATRIX_SIZE),
           block=block, grid=grid)

end.record()
end.synchronize()
print("GPU Time: %.3f ms" % start.time_till(end))

# Retrieve results only when needed
c_cpu = c_gpu.get()

print(c_cpu)
