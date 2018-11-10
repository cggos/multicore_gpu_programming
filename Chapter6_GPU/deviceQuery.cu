/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc deviceQuery.cu -o deviceQuery
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

int main ()
{
  int deviceCount = 0;
  cudaGetDeviceCount (&deviceCount);
  if (deviceCount == 0)
    printf ("No CUDA compatible GPU.\n");
  else
    {
      cudaDeviceProp pr;
      for (int i = 0; i < deviceCount; i++)
        {
          cudaGetDeviceProperties (&pr, i);
          printf ("Dev #%i is %s\n", i, pr.name);
        }
    }
  return 1;
}
