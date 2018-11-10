/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : February 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc memcpyTest.cu -o memcpyTest
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

const int MAXDATASIZE = 1024 * 1024;

int main (int argc, char **argv)
{
  int iter = atoi (argv[1]);
  int step = atoi (argv[2]);
  cudaStream_t str;
  int *h_data, *d_data;
  int i, dataSize;;
  cudaEvent_t startT, endT;
  float duration;

  cudaMallocHost ((void **) &h_data, sizeof (int) * MAXDATASIZE);
  cudaMalloc ((void **) &d_data, sizeof (int) * MAXDATASIZE);
  for (i = 0; i < MAXDATASIZE; i++)
    h_data[i] = i;

  cudaEventCreate (&startT);
  cudaEventCreate (&endT);
  cudaStreamCreate (&str);

  for (dataSize = 0; dataSize <= MAXDATASIZE; dataSize += step)
    {
      cudaEventRecord (startT, str);
      for (i = 0; i < iter; i++)
        {
          cudaMemcpyAsync (d_data, h_data, sizeof (int) * dataSize, cudaMemcpyHostToDevice, str);
        }
      cudaEventRecord (endT, str);
      cudaEventSynchronize (endT);
      cudaEventElapsedTime (&duration, startT, endT);
      printf ("%i %f\n", (int) (dataSize * sizeof (int)), duration / iter);
    }

  cudaStreamDestroy (str);
  cudaEventDestroy (startT);
  cudaEventDestroy (endT);

  cudaFreeHost (h_data);
  cudaFree (d_data);
  cudaDeviceReset ();
  return 1;
}
