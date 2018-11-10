/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Maximum number of bins are used
                 warpSize is assumed to be fixed to 32
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda.h>
#include "../common/pgm.h"

const int BINS = 256;
const int BLOCKSIZE = 192;
const int MAXPIXELSPERTHREAD = 255; // to avoid overflowing a byte counter
const int BINS4ALL = BINS * BLOCKSIZE;

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
__global__ void GPU_histogram_V2 (int *in, int N, int *h)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;
  int GRIDSIZE = gridDim.x * blockDim.x;
  __shared__ unsigned char localH[BINS4ALL];
  int bankID = locID;
  int i;

  // initialize the local, shared-memory bins
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    localH[i] = 0;

  // wait for all warps to complete the previous step
  __syncthreads ();

  //start processing the image data
  unsigned char *mySharedBank = localH + bankID;
  for (i = gloID; i < N; i += GRIDSIZE)
      {
        int temp = in[i];
        int v = temp & 0xFF;
        int v2 = (temp >> 8) & 0xFF;
        int v3 = (temp >> 16) & 0xFF;
        int v4 = (temp >> 24) & 0xFF;
        mySharedBank[v * BLOCKSIZE]++;  
        mySharedBank[v2 * BLOCKSIZE]++;
        mySharedBank[v3 * BLOCKSIZE]++;
        mySharedBank[v4 * BLOCKSIZE]++;
      }

  // wait for all warps to complete the local calculations, before updating the global counts
  __syncthreads ();

  // use atomic operations to add the local findings to the global memory bins 
  for (i = locID; i < BINS4ALL; i += blockDim.x)
    atomicAdd (h + (i/BLOCKSIZE), localH[i]); 
}

//*****************************************************************
int main (int argc, char **argv)
{

  PGMImage inImg (argv[1]);

  int *d_in, *h_in;
  int *d_hist, *h_hist, *cpu_hist;
  int i, N, bins;

  h_in = (int *) inImg.pixels;
  N = ceil ((inImg.x_dim * inImg.y_dim) / 4.0);

  bins = inImg.num_colors + 1;
  h_hist = (int *) malloc (bins * sizeof (int));
  cpu_hist = (int *) malloc (bins * sizeof (int));

  CPU_histogram (inImg.pixels, inImg.x_dim * inImg.y_dim, cpu_hist, bins);

  // ensures that shared memory/L1 cache are split in 48KB/16KB configuration
  cudaDeviceSetCacheConfig( cudaFuncCachePreferShared);

  // timing related definitions  
  cudaStream_t str;
  cudaEvent_t startT, endT;
  float duration;

  // initialize two events
  cudaStreamCreate (&str);
  cudaEventCreate (&startT);
  cudaEventCreate (&endT);

  cudaMalloc ((void **) &d_in, sizeof (int) * N);
  cudaMalloc ((void **) &d_hist, sizeof (int) * bins);
  cudaMemcpy (d_in, h_in, sizeof (int) * N, cudaMemcpyHostToDevice);
  cudaMemset (d_hist, 0, bins * sizeof (int));

  int blocks = (int)ceil(N*4.0/(BLOCKSIZE * MAXPIXELSPERTHREAD));
    
  cudaEventRecord (startT, str);
  GPU_histogram_V2 <<< blocks, BLOCKSIZE, 0, str >>> (d_in, N, d_hist);
  cudaEventRecord (endT, str);

  // wait for endT event to take place
  cudaEventSynchronize (endT);

  cudaMemcpy (h_hist, d_hist, sizeof (int) * bins, cudaMemcpyDeviceToHost);

  for (i = 0; i < BINS; i++)
    printf ("%i %i %i\n", i, cpu_hist[i], h_hist[i]);

  for (i = 0; i < BINS; i++)
    if (cpu_hist[i] != h_hist[i])
      printf ("Calculation mismatch (static) at : %i\n", i);

// calculate elapsed time
  cudaEventElapsedTime (&duration, startT, endT);
  printf ("Kernel executed for %f ms\n", duration);

// clean-up allocated objects and reset device
  cudaStreamDestroy (str);
  cudaEventDestroy (startT);
  cudaEventDestroy (endT);

  cudaFree ((void *) d_in);
  cudaFree ((void *) d_hist);
  free (h_hist);
  free (cpu_hist);
  cudaDeviceReset ();

  return 0;
}
