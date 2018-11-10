/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : January 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc odd_withErrorChecking.cu -o odd_withErrorChecking
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

#define MAXVALUE 10000

//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//------------------------------------

__global__ void countOdds (int *d, int N, int *odds)
{
  extern __shared__ int count[];

  int myID = blockIdx.x * blockDim.x + threadIdx.x;
  int localID = threadIdx.x;
  count[localID] = 0;
  if (myID < N)
    count[localID] = (d[myID] % 2);
  __syncthreads ();

  // reduction phase: sum up the block
  int step = 1;
  while (((localID | step) < blockDim.x) && ((localID & step) == 0))
    {
      count[localID] += count[localID | step];
      step *= 2;
      __syncthreads ();
    }
    
  // add to global counter
  if (localID == 0)
    atomicAdd (odds, count[0]);
}

//------------------------------------
int sharedSize (int b)
{
  return b * sizeof (int);
}

//------------------------------------

int main (int argc, char **argv)
{
  int N = atoi (argv[1]);

  int *ha, *hres, *da, *dres;   // host (h*) and device (d*) pointers

  ha = new int[N];
  hres = new int[1];

  CUDA_CHECK_RETURN (cudaMalloc ((void **) &da, sizeof (int) * N));
  CUDA_CHECK_RETURN (cudaMalloc ((void **) &dres, sizeof (int) * 1));

  numberGen (N, MAXVALUE, ha);

  CUDA_CHECK_RETURN (cudaMemcpy (da, ha, sizeof (int) * N, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN (cudaMemset (dres, 0, sizeof (int)));

  int blockSize, gridSize;
  cudaOccupancyMaxPotentialBlockSizeVariableSMem (&gridSize, &blockSize, (void *) countOdds, sharedSize, N);

  gridSize = ceil (1.0 * N / blockSize);
  printf ("Grid : %i    Block : %i\n", gridSize, blockSize);
  countOdds <<< gridSize, blockSize, blockSize * sizeof (int) >>> (da, N, dres);

  CUDA_CHECK_RETURN (cudaMemcpy (hres, dres, sizeof (int), cudaMemcpyDeviceToHost));

  // correctness check
  int oc = 0;
  for (int i = 0; i < N; i++)
    if (ha[i] % 2)
      oc++;

  printf ("%i %i\n", *hres, oc);

  CUDA_CHECK_RETURN (cudaFree ((void *) da));
  CUDA_CHECK_RETURN (cudaFree ((void *) dres));
  delete[]ha;
  delete[]hres;
  CUDA_CHECK_RETURN (cudaDeviceReset ());

  return 0;
}
