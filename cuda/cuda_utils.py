import ctypes
from typing import Tuple, List

cuda = ctypes.CDLL('libcuda.so')
nvrtc = ctypes.CDLL('libnvrtc.so')

# CUDA driver API types and constants
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_void_p

# NVRTC types
nvrtcProgram = ctypes.c_void_p
nvrtcResult = ctypes.c_int

CUDA_ERRORS = {
    0: "CUDA_SUCCESS",
    1: "CUDA_ERROR_INVALID_VALUE",
    2: "CUDA_ERROR_OUT_OF_MEMORY",
    3: "CUDA_ERROR_NOT_INITIALIZED",
    4: "CUDA_ERROR_DEINITIALIZED",
    5: "CUDA_ERROR_PROFILER_DISABLED",
    6: "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
    7: "CUDA_ERROR_PROFILER_ALREADY_STARTED",
    8: "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
    17: "CUDA_ERROR_INVALID_CONTEXT",
    18: "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
    30: "CUDA_ERROR_INVALID_HANDLE",
    33: "CUDA_ERROR_INVALID_IMAGE",
    34: "CUDA_ERROR_INVALID_CONTEXT",
    35: "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
    37: "CUDA_ERROR_MAP_FAILED",
    38: "CUDA_ERROR_UNMAP_FAILED",
    39: "CUDA_ERROR_ARRAY_IS_MAPPED",
    43: "CUDA_ERROR_INVALID_SOURCE",
    44: "CUDA_ERROR_FILE_NOT_FOUND",
    45: "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
    46: "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
    50: "CUDA_ERROR_INVALID_PTX",
    51: "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
    52: "CUDA_ERROR_NVLINK_UNCORRECTABLE",
    100: "CUDA_ERROR_NO_DEVICE",
    101: "CUDA_ERROR_INVALID_DEVICE",
    201: "CUDA_ERROR_INVALID_SOURCE",
    209: "CUDA_ERROR_ILLEGAL_ADDRESS",
    214: "CUDA_ERROR_LAUNCH_FAILED",
    215: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    216: "CUDA_ERROR_LAUNCH_TIMEOUT",
    217: "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
    218: "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
    219: "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
    700: "CUDA_ERROR_ILLEGAL_ADDRESS",
    701: "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    702: "CUDA_ERROR_INVALID_PTX",
    703: "CUDA_ERROR_LAUNCH_TIMEOUT",
    999: "CUDA_ERROR_UNKNOWN"
}

# TODO: generalize this to support multiple kernels
class CudaManager:
  def __init__(self):
    self.ctx = CUcontext()
    self.module = None
    self.init_cuda()

  def __del__(self):
    if self.module: self.check_cuda(cuda.cuModuleUnload(self.module), "cuModuleUnload")
    self.check_cuda(cuda.cuCtxDestroy(self.ctx), "cuCtxDestroy")

  @staticmethod
  def check_cuda(result: int, func_name: str = ""):
    """Checks if CUDA function call was successful. Raises RuntimeError if not."""

    if result != 0:
      err_msg = CUDA_ERRORS.get(result, f"Unknown error code {result}")
      raise RuntimeError(f"[CUDA ERROR] {func_name} failed: {err_msg} (code {result})")

  @staticmethod
  def load_kernel(file_path: str) -> str:
    """Reads a kernel file and returns its contents as a string."""

    with open(file_path, 'r') as f:
      kernel_code = f.read()
    return kernel_code

  def check_nvrtc(self, result: int, func_name: str):
    """Checks if NVRTC function call was successful. Raises RuntimeError if not."""

    if result != 0:
      log_size = ctypes.c_size_t()
      nvrtc.nvrtcGetProgramLogSize(self.program, ctypes.byref(log_size))
      log = ctypes.create_string_buffer(log_size.value)
      nvrtc.nvrtcGetProgramLog(self.program, log)
      raise RuntimeError(f"[NVRTC ERROR] {func_name} failed with code {result}:\n{log.value.decode()}")

  def  init_cuda(self):
    """Gets CUDA device and context, then initializes CUDA driver API."""

    print("[CudaManager] Initializing...")
    self.check_cuda(cuda.cuInit(0), "cuInit")
    device = CUdevice() 
    self.check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    self.check_cuda(cuda.cuCtxCreate(ctypes.byref(self.ctx), 0, device), "cuCtxCreate")

  def compile_kernel(self, src: str, kernel_name: str):
    print(f"[CudaManager] Compiling kernel {kernel_name}")
    program = nvrtcProgram()
    nvrtc.nvrtcCreateProgram.restype = nvrtcResult
    nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                ctypes.c_char_p(src.encode()),
                ctypes.c_char_p(f"{kernel_name.decode()}.cu".encode()),
                0,
                None,
                None
            )

    # compile to PTX
    opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
    # nvrtc.nvrtcCompileProgram(program, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    compile_result = nvrtc.nvrtcCompileProgram(program, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    self.check_nvrtc(compile_result, "nvrtcCompileProgram")

    # get PTX code
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size))
    ptx = (ctypes.c_char * ptx_size.value)()
    nvrtc.nvrtcGetPTX(program, ptx)

    # load PTX module
    self.module = CUmodule()
    self.check_cuda(cuda.cuModuleLoadData(ctypes.byref(self.module), ptx), "cuModuleLoadData")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    # get kernel function
    self.kfunc = CUfunction()
    self.check_cuda(cuda.cuModuleGetFunction(ctypes.byref(self.kfunc), self.module, ctypes.c_char_p(kernel_name)), "cuModuleGetFunction")
    print("[CudaManager] Kernel function pointer:", self.kfunc)

  def cuda_malloc(self, size: int) -> CUdeviceptr:
    """Allocates device memory and returns a pointer to it."""

    ptr = CUdeviceptr()
    self.check_cuda(cuda.cuMemAlloc(ctypes.byref(ptr), size), "cuMemAlloc")
    return ptr

  def cuda_free(self, ptr: CUdeviceptr):
    """Frees device memory pointed to by ptr."""
    self.check_cuda(cuda.cuMemFree(ptr), "cuMemFree")

  def memcpy_htod(self, dst: CUdeviceptr, src: ctypes.c_void_p, size: int):
    """Copies data from host to device memory."""
    self.check_cuda(cuda.cuMemcpyHtoD(dst, ctypes.c_void_p(src), size), "cuMemcpyHtoD")

  def memcpy_dtoh(self, dst: ctypes.c_void_p, src: CUdeviceptr, size):
    """Copies data from device to host memory."""
    self.check_cuda(cuda.cuMemcpyDtoH(ctypes.c_void_p(dst), src, size), "cuMemcpyDtoH")

  def launch_kernel(
      self,
      kfunc: CUfunction,
      grid: Tuple[int, int, int],
      block: Tuple[int, int, int],
      args: List[ctypes.c_void_p]
    ):
    """Launches a CUDA kernel with the given grid and block dimensions and arguments."""

    print(f"[CudaManager] Launching kernel {kfunc} with grid {grid} and block {block}")
    arg_buff = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    self.check_cuda(cuda.cuLaunchKernel(
      kfunc,
      grid[0], grid[1], grid[2],      # grid dimensions
      block[0], block[1], block[2],   # block dimensions
      0, 0,                           # shared mem and stream
      arg_buff, 0
    ), "cuLaunchKernel")
