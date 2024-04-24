%%cu
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h> 
#define N 1000

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
    // Matrix dimensions
    int width = N;
    int size = width * width;

    // Host matrices
    int *h_a, *h_b, *h_c;

    // Device matrices
    int *d_a, *d_b, *d_c;

    // Allocate memory for host matrices
    h_a = (int*)malloc(size * sizeof(int));
    h_b = (int*)malloc(size * sizeof(int));
    h_c = (int*)malloc(size * sizeof(int));

    // Initialize host matrices
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Check for memory allocation errors
    if (h_a == nullptr || h_b == nullptr || h_c == nullptr || d_a == nullptr || d_b == nullptr || d_c == nullptr) {
        std::cerr << "Failed to allocate memory\n";
        return -1;
    }

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, width);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << h_c[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}


%%cu 
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

// int main() {
//     int N = 128;
//     int size = N * N * sizeof(int);
//     int* A, * B, * C;
//     int* dev_A, * dev_B, * dev_C;

//     A = (int*)malloc(size);
//     B = (int*)malloc(size);
//     C = (int*)malloc(size);

//     cudaMalloc(&dev_A, size);
//     cudaMalloc(&dev_B, size);
//     cudaMalloc(&dev_C, size);

//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             A[i * N + j] = i * N + j;
//             B[i * N + j] = j * N + i;
//         }
//     }

//     cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

//     dim3 dimBlock(16, 16);
//     dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

//     matmul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, N);

//     cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

//     for (int i = 0; i < 10; i++) {
//         for (int j = 0; j < 10; j++) {
//             std::cout << C[i * N + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     cudaFree(dev_A);
//     cudaFree(dev_B);
//     cudaFree(dev_C);

//     free(A);
//     free(B);
//     free(C);

//     return 0;
// }
