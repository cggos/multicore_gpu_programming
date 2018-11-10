/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0.1
 Last modified : December 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc sharedDynamic.cu -o sharedDynamic
                For Ocelot use : nvcc -arch=sm_20 sharedDynamic.cu -cuda -o sharedDynamic.cu.cpp; g++ -c sharedDynamic.cu.cpp; g++ -o sharedDynamic sharedDynamic.cu.o -lglut -locelot -lGLEW -lGLU -L/usr/l/checkout/gpuocelot/ocelot/build_local/lib/
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

       
__global__ void foo(int *arraySizes)
{
   int K,L,M;
   extern __shared__ int a[];
   double *b;
   unsigned int *c;

   K = arraySizes[0];
   L = arraySizes[1];
   M = arraySizes[2];
   
   b = (double *)(&a[K]);
   c = (unsigned int *)(&b[L]);
}

int main (void)
{
  int K=100, L=20, M=15;
  int ha[3]={K,L,M};
  int *da;

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * 3));
  CUDA_CHECK_RETURN (cudaMemcpy(da, ha, sizeof (int) * 3, cudaMemcpyHostToDevice));
  foo<<< 1, 256, K*sizeof(int) + L*sizeof(double) + M*sizeof(unsigned int) >>>(da);

  
  CUDA_CHECK_RETURN (cudaFree ((void *) da));
  CUDA_CHECK_RETURN (cudaDeviceReset ());

  return 0;
}
