#!/usr/bin/env python3
import ctypes
import numpy as np

libopencl = ctypes.cdll.LoadLibrary("libOpenCL.so")

def check_error(error, msg):
  if error != 0: raise RuntimeError(f"Error in {msg}: {error}")


# OpenCL constants
CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 0x2
CL_CONTEXT_PLATFORM = 0x1084
CL_MEM_READ_ONLY = 0x10
CL_MEM_WRITE_ONLY = 0x20
CL_MEM_COPY_HOST_PTR = 0x1000
CL_PROGRAM_BUILD_LOG = 0x1183

# TODO: this assumes square matrices
kernel_src = b"""
__kernel void matMul(
  __global float* A,
  __global float* B,
  __global float* C,
  const unsigned int N
){
  int row = get_global_id(1);
  int col = get_global_id(0);

  float sum = 0.0;
  for(int i=0; i < N; i++){
    sum += A[row * N + i] * B[i * N + col];
  }
  C[row * N + col] = sum;
}
"""

N = 4
size = N * N
A = np.arange(1, size + 1, dtype=np.float32)
B = np.arange(1, size + 1, dtype=np.float32) / 2
C = np.zeros(size, dtype=np.float32)

# Init OpenCL
# get platform
platforms = ctypes.c_void_p()
num_platforms = ctypes.c_uint()
check_error(libopencl.clGetPlatformIDs(1, ctypes.byref(platforms), ctypes.byref(num_platforms)), "clGetPlatformIDs")
print(f"[+] Found {num_platforms.value} OpenCL platforms")
print(f"[*] Using platform: {platforms}")

# select device
devices = ctypes.c_void_p()
check_error(libopencl.clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, ctypes.byref(devices), None), "clGetDeviceIDs")
print("[+] Device OK")

context_properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platforms, 0)
context = libopencl.clCreateContext(
  context_properties,
  1,
  ctypes.byref(devices),
  None,
  None,
  ctypes.byref(ctypes.c_int())
)
queue = libopencl.clCreateCommandQueue(context, devices, 0, ctypes.byref(ctypes.c_int()))

# create buffers
buf_A = libopencl.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, A.nbytes, A.ctypes.data, ctypes.byref(ctypes.c_int()))
buf_B = libopencl.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, B.nbytes, B.ctypes.data, ctypes.byref(ctypes.c_int()))
buf_C = libopencl.clCreateBuffer(context, CL_MEM_WRITE_ONLY, C.nbytes, None, ctypes.byref(ctypes.c_int()))

# create program and build
program = libopencl.clCreateProgramWithSource(context, 1, ctypes.byref(ctypes.c_char_p(kernel_src)), None, ctypes.byref(ctypes.c_int()))
check_error(libopencl.clBuildProgram(program, 1, ctypes.byref(devices), None, None, None), "clBuildProgram")

kernel = libopencl.clCreateKernel(program, b"matMul", ctypes.byref(ctypes.c_int()))

# set kernel function args
libopencl.clSetKernelArg(kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(buf_A))
libopencl.clSetKernelArg(kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(buf_B))
libopencl.clSetKernelArg(kernel, 2, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(buf_C))
libopencl.clSetKernelArg(kernel, 3, ctypes.sizeof(ctypes.c_uint), ctypes.byref(ctypes.c_uint(N)))

# set global and local work size
global_size = (ctypes.c_size_t * 2)(N, N)
local_size = (ctypes.c_size_t * 2)(2, 2)

# read results back to host
check_error(libopencl.clEnqueueReadBuffer(queue, buf_C, CL_SUCCESS, 0, C.nbytes, C.ctypes.data, 0, None, None), "clEnqueueReadBuffer")

# finish queue
libopencl.clFinish(queue)

# print results
C = C.reshape((N, N))
print("Result Matrix C:")
print(C)

# cleanup
libopencl.clReleaseKernel(kernel)
libopencl.clReleaseProgram(program)
libopencl.clReleaseMemObject(buf_A)
libopencl.clReleaseMemObject(buf_B)
libopencl.clReleaseMemObject(buf_C)
libopencl.clReleaseCommandQueue(queue)
libopencl.clReleaseContext(context)
