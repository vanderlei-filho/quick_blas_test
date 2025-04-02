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

  // Time the manual matrix-vector multiplication
  clock_t start = clock();

  // Perform y = alpha * A * x + beta * y (manual computation)
  for (int i = 0; i < m; i++) {
    y[i] = 0.0; // Reset y
    for (int j = 0; j < n; j++) {
      y[i] += A[i * n + j] * x[j]; // Matrix-vector multiplication
    }
    y[i] *= alpha;
    y[i] += beta; // Apply alpha and beta
  }

  clock_t end = clock();

  printf("Manual execution time: %lf seconds\n",
         (double)(end - start) / CLOCKS_PER_SEC);

  // Free memory
  free(A);
  free(x);
  free(y);

  return 0;
}
