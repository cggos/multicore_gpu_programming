/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc hello3.cu -o hello3 -arch=sm_20
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

__global__ void hello ()
{
  int myID = ( blockIdx.z * gridDim.x * gridDim.y  + 
               blockIdx.y * gridDim.x + 
               blockIdx.x ) * blockDim.x * blockDim.y * blockDim.z + 
               threadIdx.z *  blockDim.x * blockDim.y + 
               threadIdx.y * blockDim.x + 
               threadIdx.x; 

  // Simplification of above 
//   int myID = ( blockIdx.z * gridDim.x * gridDim.y  + 
//                blockIdx.y * gridDim.x + 
//                blockIdx.x ) * blockDim.x + 
//                threadIdx.x; 

 int myID2 = threadIdx.x + blockIdx.x * blockDim.x + 
             +(blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x+
             (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * blockDim.x * gridDim.y * blockDim.y;
	       
  printf ("Hello world from %i %i\n", myID, myID2);
}

int main ()
{
  dim3 g (4, 3, 2);
  hello <<< g, 10 >>> ();
  cudaThreadSynchronize ();
  return 0;
}
