/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc memcpyTestCallback.cu -o memcpyTestCallback
 ============================================================================
 */
#include <stdio.h>
#include <time.h>
#include <cuda.h>

const int MAXDATASIZE = 1024 * 1024;
//---------------------------------------------------------
void myCallBack (cudaStream_t stream, cudaError_t status, void *userData)
{
  float *t = (float *) userData;
  clock_t x = clock();
  *t = x*1.0/CLOCKS_PER_SEC;
}
//---------------------------------------------------------
int main (int argc, char **argv)
{
  int iter = atoi (argv[1]);
  int step = atoi (argv[2]);
  cudaStream_t str;
  int *h_data, *d_data;
  int i, dataSize;;

  cudaStreamCreate(&str);
  cudaMallocHost ((void **) &h_data, sizeof (int) * MAXDATASIZE);
  cudaMalloc ((void **) &d_data, sizeof (int) * MAXDATASIZE);
  for (i = 0; i < MAXDATASIZE; i++)
    h_data[i] = i;

  float t1, t2;
  cudaStreamAddCallback (str, myCallBack, (void *) &t1, 0);
  for (dataSize = 0; dataSize <= MAXDATASIZE; dataSize += step)
    {
      for (i = 0; i < iter; i++)
        {
          cudaMemcpyAsync (d_data, h_data, sizeof (int) * dataSize, cudaMemcpyHostToDevice, str);
        }
      cudaStreamAddCallback (str, myCallBack, (void *) &t2, 0);
      cudaStreamSynchronize(str);
      printf ("%i %f\n", (int) (dataSize * sizeof (int)), (t2 - t1) / iter);
      t1 = t2;
    }

  cudaStreamDestroy (str);

  cudaFreeHost (h_data);
  cudaFree (d_data);
  cudaDeviceReset ();
  return 1;
}
