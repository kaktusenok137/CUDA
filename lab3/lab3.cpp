// Компиляция:
// nvcc lab3.cpp -o lab3 -lcublas -Xcompiler "-fopenmp"

// Запуск:
// ./lab3

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define N 3
#define IDX2C(i,j,N) (((j)*(N))+(i))

int main(int argc,char **argv){
 
  //объявление матриц
  float *A,*B,*F,*D,*C;
  float *dev_A,*dev_B,*dev_F,*dev_D,*dev_C;
  
  // выделение памяти на CPU 
  A=(float*)malloc(N*N*sizeof(float));
  B=(float*)malloc(N*N*sizeof(float));
  F=(float*)malloc(N*N*sizeof(float));
  D=(float*)malloc(N*N*sizeof(float));
  C=(float*)malloc(N*N*sizeof(float));
  
  //инициализация матриц
  
  for (int j = 0; j < N; j++)
  {
	for (int i = 0; i < N; i++)
	{
		A[IDX2C(i, j, N)]=i+j*0.007;
		B[IDX2C(i, j, N)]=2*i-3*j / 1.00897;
		F[IDX2C(i, j, N)]=2*i+j/ (-0.05);
	}
  }
  
/*

  printf("A\n");   
  for (int i = 0; i < N; i++)
  {
	for (int j = 0; j < N; j++)
	{
		printf("%f\t", A[IDX2C(i, j, N)]);
	}
	printf("\n");
  }
          
  printf("B\n");
  for (int i = 0; i < N; i++)
  {
	for (int j = 0; j < N; j++)
	{
		printf("%f\t", B[IDX2C(i, j, N)]);
	}
	printf("\n");
  }

  printf("F\n");
  for (int i = 0; i < N; i++)
  {
	for (int j = 0; j < N; j++)
	{
		printf("%f\t", F[IDX2C(i, j, N)]);
	}
	printf("\n");
  }

*/
  cudaError_t cudaStat;
  cublasStatus_t cublasStatus;
    
  //Очистка GPU
  cudaDeviceReset();
    
  //выделение памяти на GPU
  cudaStat = cudaMalloc ((void**) &dev_A,N*N*sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf ("X_GPU device memory allocation failed\n");
    return 1;
  }
    
  cudaStat = cudaMalloc ((void**) &dev_B,N*N*sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf ("X_GPU device memory allocation failed\n");
    return 1;
  }

  cudaStat = cudaMalloc ((void**) &dev_F,N*N*sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf ("X_GPU device memory allocation failed\n");
    return 1;
  }

  cudaStat = cudaMalloc ((void**) &dev_D,N*N*sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf ("X_GPU device memory allocation failed\n");
    return 1;
  }
 
  cudaStat = cudaMalloc ((void**) &dev_C,N*N*sizeof(float));
  if (cudaStat != cudaSuccess) {
    printf ("X_GPU device memory allocation failed\n");
    return 1;
  }

      
  //Копирование данных в память GPU
  cudaMemcpy(dev_A, A,N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B,N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_F, F,N*N*sizeof(float), cudaMemcpyHostToDevice);

  //контекст
  cublasHandle_t h;
  //инициализируем контекст (т.е. содержимое самой библиотеки)
  cublasCreate(&h);
  
  //Умножение матриц C=A*B*F
  const float alpha = 1.0f;
  const float beta = 0.0f;

  double time_sec = omp_get_wtime();

  //cublas<t>gemm() функция умножения матриц вида D = alpha*op(A)op(B) + beta*(C)
  //S - тип float
  cublasStatus = cublasSgemm(h,
			CUBLAS_OP_N,CUBLAS_OP_N,
			N, N, N,
			&alpha,      
			dev_A, N,       
			dev_B, N,  
			&beta,  
			dev_D, N);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
  {
	printf ("ERROR A x B = %d i=%d \n", cublasStatus,0);
	return 1;
  }

  cublasStatus = cublasSgemm(h,
			CUBLAS_OP_N,CUBLAS_OP_N,
			N, N, N,
			&alpha,      
			dev_D, N,       
			dev_F, N,  
			&beta,  
			dev_C, N);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
  {
	printf ("ERROR D x F = %d i=%d \n", cublasStatus,0);
	return 1;
  }

  //копирование результата из GPU в CPU	    
  cudaMemcpy(C, dev_C,N*N*sizeof(float), cudaMemcpyDeviceToHost);

  time_sec = omp_get_wtime() - time_sec;

/*
  printf("C=A*B*F\n");
  for (int i = 0; i < N; i++)
  {
	for (int j = 0; j < N; j++)
	{
		printf("%f\t", C[IDX2C(i, j, N)]);
        }
        printf("\n");
  }
*/

  printf("Size = %i\nTime = %f\n", N,time_sec);

  //уничтожение контекста
  cublasDestroy(h);
    
  //Освобождение памяти на GPU
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_F);
  cudaFree(dev_D);
  cudaFree(dev_C);
    
  //Освобождение памяти на CPU 
  delete [] A;
  delete [] B;
  delete [] F;
  delete [] D;
  delete [] C;

}
