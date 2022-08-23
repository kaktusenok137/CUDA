#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

//размер матрицы
#define N 3

#define block_size 1024

__global__ void mult(float* U, float* L, float* D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //for(int i=0; i<N; i++)
    while (i < N) {
	for (int j = 0; j < N; j++)
            for (int t = 0; t < N; t++)
                D[i * N + j] += U[i * N + t] * L[t * N + j];

        i += gridDim.x * blockDim.x;
    }
}


__global__ void tr(float* D, float *sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < N) {
	for (int j = 0; j < N; j++) {
		if(i==j) *sum+=D[i * N + j];
	}
	i += gridDim.x * blockDim.x;
    }

}

int main()
{
    float* U, * L, * D, * d_U, * d_L, * d_D;
    float sum=0.0f, *d_sum;
    U = (float*)malloc(N * N * sizeof(float));
    L = (float*)malloc(N * N * sizeof(float));
    D = (float*)malloc(N * N * sizeof(float));
    
    cudaMalloc((void**)&d_U, N * N * sizeof(float));
    cudaMalloc((void**)&d_L, N * N * sizeof(float));
    cudaMalloc((void**)&d_D, N * N * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));
    
    memset(D, 0, N * N * sizeof(float));
    cudaMemset(d_D, 0, N * N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) 
        {
            if (i > j) U[i * N + j] = 0;
            else U[i * N + j] = -50 + rand() % 100;
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i < j) L[i * N + j] = 0;
            else L[i * N + j] = -50 + rand() % 100;
        }
    }

    printf("U:\n");
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            printf("%f\t", U[i*N+j]);
        }
        printf("\n");
    }

    printf("\nL:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f\t", L[i * N + j]);
        }
        printf("\n");
    }

    cudaMemcpy(d_U, U, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((N + block_size - 1) / block_size);

    double time_sec = omp_get_wtime();

    mult <<< grid, block >>> (d_U, d_L, d_D);
    cudaMemcpy(D, d_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    tr <<< grid, block >>> (d_D, d_sum);

    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    time_sec = omp_get_wtime() - time_sec;

    printf("\nMatrix D (D=U*L)\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", D[i*N+j]);
        }
        printf("\n");
    }

    printf("\nThe sum of the diagonal elements of the matrix D = %f\n",sum);

    printf("\nSize = %i\nTime = %f\n", N, time_sec);

    free(U);
    free(L);
    free(D);

    cudaFree(d_U);
    cudaFree(d_L);
    cudaFree(d_D);
    cudaFree(d_sum);

    return 0;
}
