/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "../common/pgm.h"

static const int BINS = 256;

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
__global__ void GPU_histogram_static (int *in, int N, int *h)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;
  int GRIDSIZE = gridDim.x * blockDim.x;
  __shared__ int localH[BINS];
  int i;

  // initialize the local, shared-memory bins
  for (i = locID; i < BINS; i += blockDim.x)
    localH[i] = 0;

  // wait for all warps to complete the previous step
  __syncthreads ();

  //start processing the image data
  for (i = gloID; i < N; i += GRIDSIZE)
    {
      int temp = in[i];
      atomicAdd (localH + (temp & 0xFF), 1);
      atomicAdd (localH + ((temp >> 8) & 0xFF), 1);
      atomicAdd (localH + ((temp >> 16) & 0xFF), 1);
      atomicAdd (localH + ((temp >> 24) & 0xFF), 1);
    }

  // wait for all warps to complete the local calculations, before updating the global counts
  __syncthreads ();

  // use atomic operations to add the local findings to the global memory bins 
  for (i = locID; i < BINS; i += blockDim.x)
    atomicAdd (h + i, localH[i]);

}

//*****************************************************************
__global__ void GPU_histogram_dynamic (int *in, int N, int *h, int bins)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  int locID = threadIdx.x;
  extern __shared__ int localH[];
  int GRIDSIZE = gridDim.x * blockDim.x;
  int i;

  // initialize the local bins
  for (i = locID; i < bins; i += blockDim.x)
    localH[i] = 0;

  // wait for all warps to complete the previous step
  __syncthreads ();

  //start processing the image data
  for (i = gloID; i < N; i += GRIDSIZE)
    {
      int temp = in[i];
      atomicAdd (localH + (temp & 0xFF), 1);
      atomicAdd (localH + ((temp >> 8) & 0xFF), 1);
      atomicAdd (localH + ((temp >> 16) & 0xFF), 1);
      atomicAdd (localH + ((temp >> 24) & 0xFF), 1);
    }

  // wait for all warps to complete the local calculations, before updating the global counts
  __syncthreads ();

  // use atomic operations to add the local findings to the global memory bins 
  for (i = locID; i < bins; i += blockDim.x)
    atomicAdd (h + i, localH[i]);
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

  cudaMalloc ((void **) &d_in, sizeof (int) * N);
  cudaMalloc ((void **) &d_hist, sizeof (int) * bins);
  cudaMemcpy (d_in, h_in, sizeof (int) * N, cudaMemcpyHostToDevice);
  cudaMemset (d_hist, 0, bins * sizeof (int));

  GPU_histogram_static <<< 16, 256 >>> (d_in, N, d_hist);
  cudaThreadSynchronize ();     // Wait for the GPU launched work to complete

  cudaMemcpy (h_hist, d_hist, sizeof (int) * bins, cudaMemcpyDeviceToHost);

  for (i = 0; i < BINS; i++)
    if (cpu_hist[i] != h_hist[i])
      printf ("Calculation mismatch (static) at : %i\n", i);


  cudaMemset (d_hist, 0, bins * sizeof (int));
  GPU_histogram_dynamic <<< 16, 256, bins * sizeof (int)>>> (d_in, N, d_hist, bins);
  cudaThreadSynchronize ();     // Wait for the GPU launched work to complete

  cudaMemcpy (h_hist, d_hist, sizeof (int) * bins, cudaMemcpyDeviceToHost);

  for (i = 0; i < BINS; i++)
    if (cpu_hist[i] != h_hist[i])
      printf ("Calculation mismatch (dynamic) at : %i\n", i);

  cudaFree ((void *) d_in);
  cudaFree ((void *) d_hist);
  free (h_hist);
  free (cpu_hist);
  cudaDeviceReset ();

  return 0;
}
