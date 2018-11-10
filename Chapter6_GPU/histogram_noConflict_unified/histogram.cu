/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : January 2015
 License       : Released under the GNU GPL 3.0
 Description   : Runs only on CC 3.0 and above devices
 To build use  : make
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include "../common/pgm.h"

const int BINS = 256;
const int BINS4ALL = BINS * 32;

//*****************************************************************
void CPU_histogram (unsigned char *in, int N, int *h, int bins)
{
  int i;
  // initialize histogram counts
  for (i = 0; i < bins; i++)
    h[i] = 0;

  // accummulate counts
  for (i = 0; i < N; i++)
    h[in[i]]++;
}

//*****************************************************************
__global__ void GPU_histogram_atomic (int *in, int N, int *h)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;
  int GRIDSIZE = gridDim.x * blockDim.x;
  __shared__ int localH[BINS4ALL];
  int bankID = locID % warpSize;
  int i;

  // initialize the local, shared-memory bins
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    localH[i] = 0;

  // wait for all warps to complete the previous step
  __syncthreads ();

  //start processing the image data
  int *mySharedBank = localH + bankID;
  if (blockDim.x > warpSize)    // if the blocksize exceeds the warpSize, it is possible multiple warps run at the same time
    for (i = gloID; i < N; i += GRIDSIZE)
      {

        int temp = in[i];
        int v = temp & 0xFF;
        int v2 = (temp >> 8) & 0xFF;
        int v3 = (temp >> 16) & 0xFF;
        int v4 = (temp >> 24) & 0xFF;
        atomicAdd (mySharedBank + (v << 5), 1);
        atomicAdd (mySharedBank + (v2 << 5), 1);
        atomicAdd (mySharedBank + (v3 << 5), 1);
        atomicAdd (mySharedBank + (v4 << 5), 1);
      }
  else
    for (i = gloID; i < N; i += GRIDSIZE)
      {

        int temp = in[i];
        int v = temp & 0xFF;
        int v2 = (temp >> 8) & 0xFF;
        int v3 = (temp >> 16) & 0xFF;
        int v4 = (temp >> 24) & 0xFF;
        mySharedBank[v << 5]++; // Optimized version of localH[bankID + v * warpSize]++
        mySharedBank[v2 << 5]++;
        mySharedBank[v3 << 5]++;
        mySharedBank[v4 << 5]++;
      }

  // wait for all warps to complete the local calculations, before updating the global counts
  __syncthreads ();

  // use atomic operations to add the local findings to the global memory bins 
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    atomicAdd (h + (i >> 5), localH[i]);        // Optimized version of atomicAdd (h + (i/warpSize), localH[i]);
}

__device__ __managed__ int hist[BINS];
//*****************************************************************
int main (int argc, char **argv)
{
  PGMImage inImg (argv[1]);

  int *in;
  int *cpu_hist;
  int i, N, bins;

  N = ceil ((inImg.x_dim * inImg.y_dim) / (sizeof (int) * 1.0));

  bins = inImg.num_colors + 1;
  cpu_hist = (int *) malloc (bins * sizeof (int));

  CPU_histogram (inImg.pixels, inImg.x_dim * inImg.y_dim, cpu_hist, bins);

  cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
  cudaMallocManaged ((void **) &in, sizeof (int) * N);

  memcpy (in, inImg.pixels, sizeof (int) * N);
  memset (hist, 0, bins * sizeof (int));
//   cudaMemcpy (in, inImg.pixels, sizeof (int) * N, cudaMemcpyHostToDevice);
//   cudaMemset (hist, 0, bins * sizeof (int));
  
  GPU_histogram_atomic <<< 16, 256 >>> (in, N, hist);
  cudaDeviceSynchronize ();     // Wait for the GPU launched work to complete

  for (i = 0; i < BINS; i++)
    printf ("%i %i %i\n", i, cpu_hist[i], hist[i]);

  for (i = 0; i < BINS; i++)
    if (cpu_hist[i] != hist[i])
      printf ("Calculation mismatch (static) at : %i\n", i);

  cudaFree ((void *) in);
  free (cpu_hist);
  cudaDeviceReset ();

  return 0;
}
