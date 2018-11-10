/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc warpFix.cu -o warpFix
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__device__ void doSmt(int x){}
__device__ void doSmtElse(int x){}
__device__ void doFinal(int x){}

const int offPow=5;

__global__ void foo ()
{
  int ID = threadIdx.x ; 
  
  // int warpID = ID / warpSize;
  // int IDprime = (ID - (warpID + 1) / 2 * warpSize)*2 + (warpID % 2);
  // above calculations implemented with bitwise operators
  int warpID = ID >> offPow;
  int IDprime = ((ID - (((warpID + 1) >> 1) << offPow )) << 1)  + (warpID & 1);
  printf ("Thread %i %i\n", ID, IDprime);

  if(warpID % 2 ==0)
  {
    doSmt(IDprime);    
  }
  else
  {
    doSmtElse(IDprime);
  }
  doFinal(IDprime);
}

int main ()
{
  
  foo <<< 1, 128 >>> ();
  cudaThreadSynchronize ();
  return 1;
}
