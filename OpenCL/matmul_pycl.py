#!/usr/bin/env python3
import time
import numpy as np
import pyopencl as cl

from cl_utils import *

# kernel_src= """
# __kernel void matmul(
#     __global const float* A, 
#     __global const float* B, 
#     __global float* C, 
#     const unsigned int N) 
# {
#     int row = get_global_id(0);
#     int col = get_global_id(1);
    
#     float sum = 0.0f;
#     for (int i = 0; i < N; i++) {
#         sum += A[row * N + i] * B[i * N + col];
#     }
#     C[row * N + col] = sum;
# }
# """

def cl_matmul(A: np.ndarray, B: np.ndarray, C: np.ndarray, N: int,
            context: cl.Context, program: cl.Program):
  a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
  b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
  c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

  global_size = ((N + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
  global_dim = (global_size, global_size)
  local_dim = (TILE_SIZE, TILE_SIZE)

  program.matmul(queue, global_dim, local_dim, a_buf, b_buf, c_buf, np.uint32(N))
  # program.matmul(queue, (N, N), None, a_buf, b_buf, c_buf, np.uint32(N))
  cl.enqueue_copy(queue, C, c_buf).wait()
  return C


if __name__ == "__main__":
  context, queue = init_cl()

  N = 1000
  A = np.random.rand(N, N).astype(np.float32)
  B = np.random.rand(N, N).astype(np.float32)
  C = np.zeros((N, N), dtype=np.float32)

  with open("matmul.cl", 'r') as f:
    kernel_src = f.read()
    print("[*] Kernel:")
    print(kernel_src)
    print()

  program = cl.Program(context, kernel_src).build()

  cl_start_time = time.time()
  C = cl_matmul(A, B, C, N, context, program)
  cl_time = time.time() - cl_start_time

  np_start_time = time.time()
  C_numpy = np.dot(A, B)
  np_time = time.time() - np_start_time

  assert np.allclose(C, C_numpy), f"Mismatch: PyOpenCL result:\n{C}\nNumPy result:\n{C_numpy}"
  print("\nA:")
  print(A)
  print("\nB:")
  print(B)
  print("\nA * B:")
  print(C)

  print()
  print(f"[~] GPU {cl_time:.4f}s")
  print(f"[~] NumPY {np_time:.4f}s")
