/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : January 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc streamTest.cu -o streamTest
 ============================================================================
 */
#include <stdio.h>
#include <cuda.h>

const int DATASIZE=1024;

__global__ void doSmt(int *data)
{
  // Simplification of above 
  int myID = ( blockIdx.z * gridDim.x * gridDim.y  + 
               blockIdx.y * gridDim.x + 
               blockIdx.x ) * blockDim.x + 
               threadIdx.x; 

  printf ("Hello world from %i\n", myID);
}

int main ()
{
  cudaStream_t str[2];
  int *h_data[2], *d_data[2];
  int i;
  
  for(i=0;i<2;i++)
  {
    cudaStreamCreate(&(str[i]));
    h_data[i] = (int *)malloc(sizeof(int) * DATASIZE);
    cudaMalloc((void ** )&(d_data[i]), sizeof(int) * DATASIZE);

    // inititalize h_data[i]....

    cudaMemcpyAsync(d_data[i], h_data[i], sizeof(int) * DATASIZE, cudaMemcpyHostToDevice, str[i]);
   
    doSmt <<< 10, 256, 0, str[i] >>> (d_data[i]);
    
    cudaMemcpyAsync(h_data[i], d_data[i], sizeof(int) * DATASIZE, cudaMemcpyDeviceToHost, str[i]);    
  }

  cudaStreamSynchronize(str[0]);
  cudaStreamSynchronize(str[1]);
  cudaStreamDestroy(str[0]);
  cudaStreamDestroy(str[1]);

  for(i=0;i<2;i++)
  {
    free(h_data[i]);
    cudaFree(d_data[i]);
  }
  cudaDeviceReset();
  return 1;
}
