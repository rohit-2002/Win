%%cu
#include <stdio.h>

#define N 1000000 // Length of the vectors
#define MATRIX_SIZE 1024 // Size of the square matrices

// Kernel function for vector addition
__global__ void vectorAdd(int *a, int *b, int *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

// Kernel function for matrix multiplication
__global__ void matrixMul(int *a, int *b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    // Addition of two large vectors
    printf("Vector Addition:\n");

    // Allocate memory for host vectors
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    h_c = (int*)malloc(N * sizeof(int));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory for device vectors
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for vector addition
    int blockSizeVector = 256;
    int numBlocksVector = (N + blockSizeVector - 1) / blockSizeVector;

    // Launch vector addition kernel
    vectorAdd<<<numBlocksVector, blockSizeVector>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify result for vector addition
    for (int i = 0; i < 10; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory for vector addition
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory for vector addition
    free(h_a);
    free(h_b);
    free(h_c);

    // Matrix Multiplication using CUDA C
    printf("\nMatrix Multiplication:\n");

    // Host matrices
    int *h_matrixA, *h_matrixB, *h_matrixC;

    // Allocate memory for host matrices
    h_matrixA = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    h_matrixB = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    h_matrixC = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    // Initialize host matrices
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        h_matrixA[i] = i;
        h_matrixB[i] = i * 2;
    }

    // Allocate memory for device matrices
    int *d_matrixA, *d_matrixB, *d_matrixC;
    cudaMalloc(&d_matrixA, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_matrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    cudaMalloc(&d_matrixC, MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(d_matrixA, h_matrixA, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for matrix multiplication
    dim3 blockSizeMatrix(16, 16);
    dim3 gridSizeMatrix((MATRIX_SIZE + blockSizeMatrix.x - 1) / blockSizeMatrix.x, (MATRIX_SIZE + blockSizeMatrix.y - 1) / blockSizeMatrix.y);

    // Launch matrix multiplication kernel
    matrixMul<<<gridSizeMatrix, blockSizeMatrix>>>(d_matrixA, d_matrixB, d_matrixC, MATRIX_SIZE);

    // Copy result back to host
    cudaMemcpy(h_matrixC, d_matrixC, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify result for matrix multiplication
    printf("Result matrix:\n");
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            printf("%d ", h_matrixC[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Free device memory for matrix multiplication
    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixC);

    // Free host memory for matrix multiplication
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixC);

    return 0;
}
