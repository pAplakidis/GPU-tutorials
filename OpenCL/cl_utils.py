import time
import numpy as np
import pyopencl as cl

TILE_SIZE = 16

def init_cl():
  init_time = time.time()
  platform = cl.get_platforms()[0]
  device = platform.get_devices()[0]
  context = cl.Context([device])
  queue = cl.CommandQueue(context)
  print(f"[+] Initialized OpenCL - platform: {platform}, device: {device}, context: {context}, queue: {queue}\tTime: {time.time() - init_time:0.2f}s")
  return context, queue

