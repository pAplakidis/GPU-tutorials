#!/usr/bin/env python3
import time
import ctypes
import numpy as np

from cuda_utils import *


if __name__ == "__main__":
  N = 10000
  block_size = 4

  cm = CudaManager()
  kernel_code = cm.load_kernel("kernels/vecadd.cu")
  cm.compile_kernel(kernel_code, b"vec_add_kernel")

  # Create host arrays with pinned memory
  A = np.random.uniform(0, 10, N).astype(np.float32)
  B = np.random.uniform(0, 10, N).astype(np.float32)
  C = np.empty(N, dtype=np.float32)

  print("A:", A)
  print("B:", B)
  print()

  # Allocate device memory
  d_A = cm.cuda_malloc(N * 4)
  d_B = cm.cuda_malloc(N * 4)
  d_C = cm.cuda_malloc(N * 4)

  # Transfer data
  cm.memcpy_htod(d_A, A.ctypes.data, N * 4)  # HostToDevice
  cm.memcpy_htod(d_B, B.ctypes.data, N * 4)

  # Prepare kernel arguments
  args = [
    ctypes.c_void_p(d_A.value),
    ctypes.c_void_p(d_B.value),
    ctypes.c_void_p(d_C.value),
    ctypes.c_int(N)
  ]

  # Launch kernel
  cuda_start_time = time.time()
  grid = ((N + block_size - 1) // block_size, 1, 1)
  block = (block_size, 1, 1)
  cm.launch_kernel(cm.kfunc, grid, block, args)

  # Copy back result
  cm.memcpy_dtoh(C.ctypes.data, d_C, N * 4)  # DeviceToHost

  print()
  print(f"CUDA Time: {time.time() - cuda_start_time:.4f}s")
  np_start_time = time.time()
  C_np = A + B
  print(f"NumPY Time: {time.time() - np_start_time:.4f}s")
  
  py_start_time = time.time()
  C_py = np.empty(N, dtype=np.float32)
  for i in range(N):
    C_py[i] = A[i] + B[i]
  print(f"Python Time: {time.time() - py_start_time:.4f}s")

  print()
  print("[numpy] Result:", C_np)
  print("[cuda] Result:", C)
  assert np.allclose(C_np, C)

  cm.cuda_free(d_A)
  cm.cuda_free(d_B)
  cm.cuda_free(d_C)
