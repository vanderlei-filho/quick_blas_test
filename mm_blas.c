#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  int m = 100000, n = 100000; // Matrix dimensions
  double alpha = 1.0, beta = 0.0;

  // Allocate memory for matrices and vectors
  double *A = (double *)malloc(m * n * sizeof(double));
  double *x = (double *)malloc(n * sizeof(double));
  double *y = (double *)malloc(m * sizeof(double));

  // Initialize A and x with some values (e.g., random values)
  for (int i = 0; i < m * n; i++) {
    A[i] = (double)(rand() % 10);
  }
  for (int i = 0; i < n; i++) {
    x[i] = 1.0; // Vector x
  }
  for (int i = 0; i < m; i++) {
    y[i] = 0.0; // Result vector y
  }

  // Time the BLAS operation
  clock_t start = clock();

  // Perform y = alpha * A * x + beta * y (using BLAS)
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, x, 1, beta, y, 1);

  clock_t end = clock();

  printf("BLAS execution time: %lf seconds\n",
         (double)(end - start) / CLOCKS_PER_SEC);

  // Free memory
  free(A);
  free(x);
  free(y);

  return 0;
}
