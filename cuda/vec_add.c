#include <stdio.h>
#include <stdlib.h>

void vec_add(float *A, float *B, float *C, int N) {
  for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int N = 10;

  float *A = (float *)malloc(N * sizeof(float));
  float *B = (float *)malloc(N * sizeof(float));
  float *C = (float *)malloc(N * sizeof(float));

  if (!(A && B && C)) {
    perror("Error allocating memory for vectors\n");
    return 1;
  }

  for (int i = 0; i < N; i++) {
    A[i] = (float)(rand() % (10 - 0 + 1) + 0);
    B[i] = (float)(rand() % (10 - 0 + 1) + 0);
    C[i] = 0;
  }

  vec_add(A, B, C, N);
  for (int i = 0; i < N; i++) {
    printf("%f\t", C[i]);
  }
  printf("\n");

  free(C);
  free(B);
  free(A);

  return 0;
}
