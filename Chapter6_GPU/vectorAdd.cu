/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc vectorAdd.cu -o vectorAdd
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

static const int BLOCK_SIZE = 256;
static const int N = 2000;

#define CUDA_CHECK_RETURN(value) {           \
    cudaError_t _m_cudaStat = value;         \
    if (_m_cudaStat != cudaSuccess) {        \
         fprintf(stderr, "Error %s at line %d in file %s\n",              \
                 cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);    \
         exit(1);                                                         \
       } }

__global__ void vadd (int *a, int *b, int *c, int N)
{
  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  if (myID < N)
    c[myID] = a[myID] + b[myID];
}

int main (void)
{
  int *ha, *hb, *hc, *da, *db, *dc;     // host (h*) and device (d*) pointers
  int i;

  ha = new int[N];
  hb = new int[N];
  hc = new int[N];

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &db, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &dc, sizeof (int) * N));

  for (i = 0; i < N; i++)
    {
      ha[i] = rand () % 10000;
      hb[i] = rand () % 10000;
    }

  CUDA_CHECK_RETURN (cudaMemcpy (da, ha, sizeof (int) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN (cudaMemcpy (db, hb, sizeof (int) * N, cudaMemcpyHostToDevice));

  int grid = ceil (N * 1.0 / BLOCK_SIZE);
  vadd <<< grid, BLOCK_SIZE >>> (da, db, dc, N);

  CUDA_CHECK_RETURN (cudaThreadSynchronize ());
  // Wait for the GPU launched work to complete
  CUDA_CHECK_RETURN (cudaGetLastError ());
  CUDA_CHECK_RETURN (cudaMemcpy (hc, dc, sizeof (int) * N, cudaMemcpyDeviceToHost));

  for (i = 0; i < N; i++)
    {
      if (hc[i] != ha[i] + hb[i])
        printf ("Error at index %i : %i VS %i\n", i, hc[i], ha[i] + hb[i]);
    }

  CUDA_CHECK_RETURN (cudaFree ((void *) da));
  CUDA_CHECK_RETURN (cudaFree ((void *) db));
  CUDA_CHECK_RETURN (cudaFree ((void *) dc));
  delete[]ha;
  delete[]hb;
  delete[]hc;
  CUDA_CHECK_RETURN (cudaDeviceReset ());

  return 0;
}
