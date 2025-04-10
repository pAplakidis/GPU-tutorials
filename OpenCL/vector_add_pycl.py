#!/usr/bin/env python3
import time
import numpy as np
from pyopencl import cl

from cl_utils import *

kernel_src= """
__kernel void vec_add(__global const float* A, 
                         __global const float* B, 
                         __global float* C, 
                         const int N) {
    int id = get_global_id(0);
    if (id < N) {
        C[id] = A[id] + B[id];
    }
}
"""

def cl_vecadd(A: np.ndarray, B: np.ndarray, C: np.ndarray, N: int,
            context: cl.Context):
  a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
  b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
  c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

  program = cl.Program(context, kernel_src).build()
  program.vec_add(queue, (N, N), None, a_buf, b_buf, c_buf, np.uint32(N))
  cl.enqueue_copy(queue, C, c_buf).wait()
  return C

if __name__ == "__main__":
  context, queue = init_cl()

  N = 1000
  A = np.random.rand(N).astype(np.float32)
  B = np.random.rand(N).astype(np.float32)
  C = np.zeros((N), dtype=np.float32)

  np_start_time = time.time()
  C_numpy = A + B
  np_time = time.time() - np_start_time

  cl_start_time = time.time()
  C = cl_vecadd(A, B, C, N, context)
  cl_time = time.time() - cl_start_time

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
