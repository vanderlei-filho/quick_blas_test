#ifdef USE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#else
  #include <cblas.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

// Function to get time in microseconds for more precise timing
double get_time_usec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// Sequential manual implementation
void sequential_matrix_vector_multiply(int m, int n, double alpha, const double *A, 
                                  const double *x, double beta, double *y) {
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

// Parallel manual implementation using OpenMP
void parallel_matrix_vector_multiply(int m, int n, double alpha, const double *A, 
                                 const double *x, double beta, double *y) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

int main() {
    int m = 1000, n = 1000; // Adjust size as needed
    double alpha = 1.0, beta = 0.0;
    int num_threads = 4;
    
    // Allocate memory for matrices and vectors
    double *A = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *y_blas = (double *)malloc(m * sizeof(double));
    double *y_seq = (double *)malloc(m * sizeof(double));
    double *y_par = (double *)malloc(m * sizeof(double));
    
    srand(42); // Use fixed seed for reproducibility
    
    // Initialize A and x with some values
    for (int i = 0; i < m * n; i++) {
        A[i] = (double)(rand() % 10) / 10.0;
    }
    for (int i = 0; i < n; i++) {
        x[i] = (double)(rand() % 10) / 10.0;
    }
    for (int i = 0; i < m; i++) {
        y_blas[i] = 0.0;
        y_seq[i] = 0.0;
        y_par[i] = 0.0;
    }
    
#ifdef USE_MKL
    // Set number of threads explicitly for MKL
    printf("Setting MKL to use %d threads\n", num_threads);
    mkl_set_num_threads(num_threads);
#endif
    
    // Set OpenMP threads for parallel implementation
    omp_set_num_threads(num_threads);
    printf("Setting OpenMP to use %d threads\n", num_threads);
    
    // Warm-up
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A, 2, x, 1, 0.0, y_blas, 1);
    
    // Time the BLAS operation
    double start_blas = get_time_usec();
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, x, 1, beta, y_blas, 1);
    double end_blas = get_time_usec();
    double blas_time = (end_blas - start_blas) / 1000000.0;
    
    // Time the sequential operation
    double start_seq = get_time_usec();
    sequential_matrix_vector_multiply(m, n, alpha, A, x, beta, y_seq);
    double end_seq = get_time_usec();
    double seq_time = (end_seq - start_seq) / 1000000.0;
    
    // Time the parallel operation
    double start_par = get_time_usec();
    parallel_matrix_vector_multiply(m, n, alpha, A, x, beta, y_par);
    double end_par = get_time_usec();
    double par_time = (end_par - start_par) / 1000000.0;
    
    // Calculate checksums
    double blas_checksum = 0.0, seq_checksum = 0.0, par_checksum = 0.0;
    for (int i = 0; i < m; i++) {
        blas_checksum += y_blas[i];
        seq_checksum += y_seq[i];
        par_checksum += y_par[i];
    }

    printf("Matrix size: %d x %d\n", m, n);
    printf("BLAS execution time: %.6f seconds (checksum: %.6f)\n", 
           blas_time, blas_checksum);
    printf("Naive-Sequential execution time: %.6f seconds (checksum: %.6f)\n", 
           seq_time, seq_checksum);
    printf("Naive-Parallel execution time: %.6f seconds (checksum: %.6f)\n", 
           par_time, par_checksum);
    
    printf("BLAS vs Naive-Sequential speedup: %.2f times\n", seq_time / blas_time);
    printf("BLAS vs Naive-Parallel speedup: %.2f times\n", par_time / blas_time);
    printf("Naive-Parallel vs Naive-Sequential speedup: %.2f times\n", seq_time / par_time);
    
    // Free memory
    free(A);
    free(x);
    free(y_blas);
    free(y_seq);
    free(y_par);
    
    return 0;
}
