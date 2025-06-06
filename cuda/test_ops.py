#!/usr/bin/env python3
import os
import ctypes
import numpy as np
from typing import Tuple, List

from cuda_utils import CudaManager

DEBUG = int(os.getenv("DEBUG", 0))


# TODO: make class (cm possibly) more generic to support multiple kernels without reallocating memory
# TODO: tensors should already be to(device) later on
class CudaBinaryOps:
  def __init__(self, tile_size: int = 16, debug: int = 1):
    self.cm = CudaManager(debug=debug)
    self.tile_size = tile_size
    self.kernel_code = None

  @staticmethod
  def flatten_tensors(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten tensors while preserving memory order."""
    A_flat = A.ravel()
    B_flat = B.ravel()
    return A_flat, B_flat

  @staticmethod
  def prep_kargs(
    d_A: ctypes.c_void_p,
    d_B: ctypes.c_void_p,
    d_C: ctypes.c_void_p,
    dim1: int,
    dim2: int,
    dim3: int
  ) -> List[ctypes.c_void_p]:
    """"Prepare kernel arguments."""
    return [
      ctypes.c_void_p(d_A.value),
      ctypes.c_void_p(d_B.value),
      ctypes.c_void_p(d_C.value),
      ctypes.c_int(dim1),
      ctypes.c_int(dim2),
      ctypes.c_int(dim3)
    ]

  @staticmethod
  def prep_kargs_conv2d(
    d_in: ctypes.c_void_p,
    d_kernel: ctypes.c_void_p,
    d_out: ctypes.c_void_p,
    input_height: int,
    input_width: int,
    kernel_height: int,
    kernel_width: int
  ) -> List[ctypes.c_void_p]:
    """Prepare kernel arguments for 2D convolution."""
    return [
      ctypes.c_void_p(d_in.value),
      ctypes.c_void_p(d_kernel.value),
      ctypes.c_void_p(d_out.value),
      ctypes.c_int(input_height),
      ctypes.c_int(input_width),
      ctypes.c_int(kernel_height),
      ctypes.c_int(kernel_width)
    ]

  def allocate_device_memory(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Tuple[ctypes.c_void_p, ctypes.c_void_p]:
    """Allocate device memory for tensors."""
    d_A = self.cm.cuda_malloc(A.nbytes)
    d_B = self.cm.cuda_malloc(B.nbytes)
    d_C = self.cm.cuda_malloc(C.nbytes)
    return d_A, d_B, d_C

  def copy_data_to_device(self, d_A: ctypes.c_void_p, d_B: ctypes.c_void_p, A_flat: np.ndarray, B_flat: np.ndarray):
    """Copy data from host to device."""
    self.cm.memcpy_htod(d_A, A_flat.ctypes.data, A_flat.nbytes)
    self.cm.memcpy_htod(d_B, B_flat.ctypes.data, B_flat.nbytes)

  def free_device_tensors(self, d_A: ctypes.c_void_p, d_B: ctypes.c_void_p, d_C: ctypes.c_void_p):
    """Free tensors from device memory."""
    self.cm.cuda_free(d_A)
    self.cm.cuda_free(d_B)
    self.cm.cuda_free(d_C)

  def add(self, A: np.ndarray, B: np.ndarray, block_size: Tuple = (8, 8, 8)) -> np.ndarray:
    """Add two homogeneous tensors of any dimension (1D, 2D, 3D) using CUDA."""
    assert A.shape == B.shape, "Tensors must have the same shape"

    self.kernel_code = self.cm.load_kernel("kernels/add.cu")
    self.cm.compile_kernel(self.kernel_code, b"add_kernel")

    dims = A.shape
    padded_dims = dims + (1,) * (3 - len(dims))  # Pad to 3D
    dim1, dim2, dim3 = padded_dims[:3]

    # tensors.to(device)
    A_flat, B_flat = self.flatten_tensors(A, B)
    C_flat = np.empty_like(A_flat)
    d_A, d_B, d_C = self.allocate_device_memory(A_flat, B_flat, C_flat)
    self.copy_data_to_device(d_A, d_B, A_flat, B_flat)

    # TODO: double check and study this
    # Define grid and block sizes
    grid = (
      (dim3 + block_size[0] - 1) // block_size[0],
      (dim2 + block_size[1] - 1) // block_size[1],
      (dim1 + block_size[2] - 1) // block_size[2],
    )

    # Kernel launch and copy result back to host
    args = self.prep_kargs(d_A, d_B, d_C, dim1, dim2, dim3)
    self.cm.launch_kernel(self.cm.kfunc, grid, block_size, args)
    self.cm.memcpy_dtoh(C_flat.ctypes.data, d_C, C_flat.nbytes)
    self.free_device_tensors(d_A, d_B, d_C)
    return C_flat.reshape(dims)

  # TODO: tensor cores
  def matmul(self, A: np.ndarray, B: np.ndarray, block_size: Tuple = (8, 8, 1)) -> np.ndarray:
    """Matrix multiplication using CUDA."""
    assert A.shape[1] == B.shape[0], "Inner dimensions must match"

    self.kernel_code = self.cm.load_kernel("kernels/matmul.cu")
    self.cm.compile_kernel(self.kernel_code, b"matmul_tiled_kernel")

    M, K = A.shape
    _, N = B.shape[1], B.shape[1]

    C = np.zeros((M, N), dtype=np.float32)
    d_A, d_B, d_C = self.allocate_device_memory(A, B, C)
    self.copy_data_to_device(d_A, d_B, A, B)

    # Define grid and block sizes
    grid = (
      (N + self.tile_size - 1) // self.tile_size,
      (M + self.tile_size - 1) // self.tile_size,
      1,
    )
    block_size = (self.tile_size, self.tile_size, 1)
    
    # Kernel launch and copy result back to host
    args = self.prep_kargs(d_A, d_B, d_C, M, N, K)
    self.cm.launch_kernel(self.cm.kfunc, grid, block_size, args)
    self.cm.memcpy_dtoh(C.ctypes.data, d_C, C.nbytes)
    self.free_device_tensors(d_A, d_B, d_C)
    return C

  def conv2d(self, input: np.array, kernel: np.array, block_size = (16, 16, 1)) -> np.array:
    assert input.ndim == 2, "Input must be a 2D array"
    assert kernel.ndim == 2, "Kernel must be a 2D array"
    assert input.dtype == np.float32, "Input must be of type float32"
    assert kernel.dtype == np.float32, "Kernel must be of type float32"
    assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1, "Kernel dimensions must be odd"
    assert kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert input.shape[0] >= kernel.shape[0] and input.shape[1] >= kernel.shape[1], "Input must be larger than or equal to kernel dimensions"

    self.kernel_code = self.cm.load_kernel("kernels/conv2d.cu")
    print(self.kernel_code)
    self.cm.compile_kernel(self.kernel_code, b"conv2d_kernel")

    input_height = input.shape[0]
    input_width = input.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output = np.zeros_like(input)
    d_in, d_kernel, d_out = self.allocate_device_memory(input, kernel, output)
    self.copy_data_to_device(d_in, d_kernel, input, kernel)
  
    grid = (
      (input_width + block_size[0] - 1) // block_size[0],
      (input_height + block_size[1] - 1) // block_size[1],
      1,
    )

    args = self.prep_kargs_conv2d(d_in, d_kernel, d_out, input_height, input_width, kernel_height, kernel_width)
    self.cm.launch_kernel(self.cm.kfunc, grid, block_size, args)
    self.cm.memcpy_dtoh(output.ctypes.data, d_out, output.nbytes)
    self.free_device_tensors(d_in, d_kernel, d_out)
    return output


if __name__ == "__main__":
  print("[*] Testing add")
  # 1D vectors
  vec1 = np.random.rand(10000).astype(np.float32)
  vec2 = np.random.rand(10000).astype(np.float32)
  result_vec = CudaBinaryOps(debug=DEBUG).add(vec1, vec2, block_size=(8, 8, 8))
  assert np.allclose(result_vec, vec1 + vec2), "1D vector addition failed"

  # 2D matrices
  mat1 = np.random.rand(512, 512).astype(np.float32)
  mat2 = np.random.rand(512, 512).astype(np.float32)
  result_mat = CudaBinaryOps(debug=DEBUG).add(mat1, mat2, block_size=(16, 16, 1))
  assert np.allclose(result_mat, mat1 + mat2), "2D matrix addition failed"

  # 3D tensors
  tensor1 = np.random.rand(64, 64, 64).astype(np.float32)
  tensor2 = np.random.rand(64, 64, 64).astype(np.float32)
  result_tensor = CudaBinaryOps(debug=DEBUG).add(tensor1, tensor2, block_size=(256, 1, 1))
  assert np.allclose(result_tensor, tensor1 + tensor2), "3D tensor addition failed"

  print("[+] Add OK\n")

  print("[*] Testing matmul")
  A = np.random.rand(1024, 512).astype(np.float32)
  B = np.random.rand(512, 2048).astype(np.float32)
  C_gpu = CudaBinaryOps(debug=DEBUG).matmul(A, B)
  C_np = np.dot(A, B)
  assert np.allclose(C_gpu, C_np, atol=1e-4)
  print("[+] Matmul OK\n")

  print("[*] Testing add after matmul")
  D = np.random.rand(1024, 2048).astype(np.float32)
  E_gpu = CudaBinaryOps(debug=DEBUG).add(C_gpu, D)
  E_np = C_gpu + D
  assert np.allclose(E_gpu, E_np, atol=1e-4)
  print("[+] Add after matmul OK\n")

  print("[*] Testing conv2d")
  CONV_IN = np.random.rand(32, 32).astype(np.float32)
  KERNEL = np.random.rand(3, 3).astype(np.float32)
  CONV_OUT = CudaBinaryOps(debug=DEBUG).conv2d(CONV_IN, KERNEL)
  print("[+] Conv2D OK\n")
