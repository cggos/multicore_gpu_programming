/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc warpFixMultiway.cu -o warpFixMultiway
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__device__ void doSmt (int x) {}
__device__ void doSmtElse (int x) {}
__device__ void doSmtElse2 (int x) {}
__device__ void doFinal (int x) {}

const int N = 3;

__global__ void foo ()
{
  int ID = threadIdx.x;
  int warpID = ID / warpSize;
  int grpOff = (warpID / N) * warpSize * N;
  int IDprime = grpOff + (ID - grpOff - (warpID % N) * warpSize) * N + (warpID % N);

  printf ("Thread %i %i\n", ID, IDprime);

  switch (warpID % N)
    {
    case 0:
      doSmt (IDprime);
      break;
    case 1:
      doSmtElse (IDprime);
      break;
    default:
      doSmtElse2 (IDprime);
    }
  doFinal (IDprime);
}

int main ()
{

  foo <<< 1, 3*32*2 >>> ();
  cudaThreadSynchronize ();
  return 1;
}
