#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// OpenCL kernel for matrix multiplication
const char *kernelSource = R"(
__kernel void matMul(
    __global float* A,
    __global float* B,
    __global float* C,
    const unsigned int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);

    float sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
)";

void checkError(cl_int err, const char *operation) {
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Error during operation '%s': %d\n", operation, err);
    exit(1);
  }
}

int main() {
  const int N = 4; // Size of the matrix
  size_t bytes = N * N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);

  // Initialize matrices
  for (int i = 0; i < N * N; i++) {
    h_A[i] = i + 1.0f;
    h_B[i] = (i + 1.0f) / 2.0f;
  }

  // Get OpenCL platform
  cl_int err;
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);
  checkError(err, "clGetPlatformIDs");
  printf("[+] Platform %d\n", platform);

  // Get OpenCL device
  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  checkError(err, "clGetDeviceIDs");

  // Create OpenCL context
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkError(err, "clCreateContext");

  // Create OpenCL command queue
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
  checkError(err, "clCreateCommandQueue");

  // Create OpenCL buffers
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "clCreateBuffer A");
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
  checkError(err, "clCreateBuffer B");
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  checkError(err, "clCreateBuffer C");

  // Write data to device
  err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, bytes, h_A, 0, NULL, NULL);
  checkError(err, "clEnqueueWriteBuffer A");
  err = clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, bytes, h_B, 0, NULL, NULL);
  checkError(err, "clEnqueueWriteBuffer B");

  // Create OpenCL program
  cl_program program =
      clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
  checkError(err, "clCreateProgramWithSource");

  // Build OpenCL program
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                          buffer, &len);
    fprintf(stderr, "Build log:\n%s\n", buffer);
    exit(1);
  }

  // Create OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "matMul", &err);
  checkError(err, "clCreateKernel");

  // Set kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  checkError(err, "clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  checkError(err, "clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
  checkError(err, "clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
  checkError(err, "clSetKernelArg 3");

  // Define global and local work size
  size_t global[2] = {N, N};
  size_t local[2] = {2, 2};

  // Execute kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL,
                               NULL);
  checkError(err, "clEnqueueNDRangeKernel");

  // Read back results
  err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C, 0, NULL, NULL);
  checkError(err, "clEnqueueReadBuffer");

  // Print results
  printf("Result Matrix C:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%f ", h_C[i * N + j]);
    }
    printf("\n");
  }

  // Clean up
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
