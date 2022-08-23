// nvcc lab2.cu -o lab2 -Xcompiler "-fopenmp"
// ./lab2

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define N 10
#define block_size 1024

__global__ void mult(float *A, float *B, float *D)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float temp[N];
    while(i < N)
    {
        for(int j=0; j<N; j++)                      
        {
            temp[i] = 0;
            //__syncthreads();
            for(int k=0; k<N; k++)                
                temp[i] += A[i*N+k]*B[k*N+j];
            
            D[i*N+j] = temp[i];
        }
        
        i += gridDim.x*blockDim.x;    
    }
}


__global__ void mult2(float *D, float *F, float *C)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float temp[N];
    while(i < N)
    {
        for(int j=0; j<N; j++)                      
        {
            temp[i] = 0;
            //__syncthreads();
            for(int k=0; k<N; k++)                
                temp[i] += D[i*N+k]*F[k*N+j];
            
            C[i*N+j] = temp[i];
        }
        
        i += gridDim.x*blockDim.x;    
    }
}


int main()
{
    float *A, *B, *F, *D, *C, *d_A, *d_B, *d_F, *d_D, *d_C;
    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*N*sizeof(float));
    F = (float*)malloc(N*N*sizeof(float));
    D = (float*)malloc(N*N*sizeof(float));
    C = (float*)malloc(N*N*sizeof(float));

    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_F, N*N*sizeof(float));
    cudaMalloc((void**)&d_D, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));

    memset(D, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));

    cudaMemset(d_D, 0, N*N*sizeof(float));
    cudaMemset(d_C, 0, N*N*sizeof(float));

    printf("Matrix A\n");
    FILE *file1 = fopen("A.txt", "r");
    if (file1 == NULL) {
        printf("File open error!");
    }
    else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fscanf(file1, "%f", &A[i*N+j]);
                printf("%f\t", A[i*N+j]);
            }
            printf("\n");
        }
    }
    fclose(file1);

    printf("\nMatrix B\n");
    FILE *file2 = fopen("B.txt", "r");
    if (file2 == NULL) {
        printf("File open error!");
    }
    else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fscanf(file2, "%f", &B[i*N+j]);
                printf("%f\t", B[i*N+j]);
            }
            printf("\n");
        }
    }
    fclose(file2);

    printf("\nMatrix F\n");
    FILE *file3 = fopen("F.txt", "r");
    if (file3 == NULL) {
        printf("File open error!");
    }
    else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fscanf(file3, "%f", &F[i*N+j]);
                printf("%f\t", F[i*N+j]);
            }
            printf("\n");
        }
    }
    fclose(file3);

    
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((N + block_size - 1) / block_size);

    double time_sec = omp_get_wtime();

    mult <<< grid, block >>> (d_A, d_B, d_D);
    cudaMemcpy(D, d_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    mult2 <<< grid, block >>> (d_D, d_F, d_C);
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    time_sec = omp_get_wtime() - time_sec;

    printf("\nMatrix C (C=A*B*F)\n");
    FILE *res = fopen("res.txt", "w+");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(res, "%f ", C[i*N+j]);
            printf("%f\t", C[i*N+j]);
        }
        fprintf(res,"\n");
        printf("\n");
    }
    fclose(res);

    printf("Size = %i\nTime = %f\n", N,time_sec);

    free(A);
    free(B);
    free(F);
    free(D);
    free(C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_F);
    cudaFree(d_D);
    cudaFree(d_C);

    return 0;
}
