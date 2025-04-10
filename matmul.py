#!/usr/bin/env python3
import random
import numpy as np

M_A = 10
N_B = 10
M_B = M_A
N_A = 10

A = [[random.random() for _ in range(N_A)] for _ in range(M_A)]
B = [[random.random() for _ in range(M_B)] for _ in range(N_B)]

def matmul(A, B, M_A, N_A, M_B, N_B):
  assert N_A == M_B
  C = [[0 for _ in range(M_A)] for _ in range(N_B)]

  for i in range(M_A):
    for j in range(N_B):
      for k in range(N_A):
        C[i][j] += A[i][k] * B[k][j]
  return C


if __name__ == "__main__":
  C_py = matmul(A, B, M_A, N_A, M_B, N_B)
  C_np = np.dot(np.array(A), np.array(B))
  assert np.allclose(np.array(C_py), C_np, atol=1e-7)
  print("[+] OK")

