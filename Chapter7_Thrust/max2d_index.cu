/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Finding the index of the most distant 2D point
 To build use  : nvcc max2d_index.cu -o max2d_index
 ============================================================================
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <stdio.h>
#include <cuda.h>

using namespace std;
//**********************************************
struct distFunct
{
  __host__ __device__ float operator () (float &x, float &y)
  {
    return x * x + y * y;
  }
};
//**********************************************
const int BLOCKSIZE = 256;
// calculate one local maximum per block and stores it in locMax, subDist
__global__ void findMax (float *dist, int N, int *locMax, float *subDist)
{
  __shared__ float maxD[BLOCKSIZE];
  __shared__ int maxIdx[BLOCKSIZE];

  int gloID = threadIdx.x + blockIdx.x * blockDim.x;
  int locID = threadIdx.x;
  if (gloID * 2 + 1 >= N)
    return;
  float d1 = dist[gloID * 2];
  float d2 = dist[gloID * 2 + 1];
  if (d1 > d2)
    {
      maxD[locID] = d1;
      maxIdx[locID] = gloID * 2;
    }
  else
    {
      maxD[locID] = d2;
      maxIdx[locID] = gloID * 2 + 1;
    }
  __syncthreads ();

  for (int phase = 1; phase < BLOCKSIZE; phase *= 2)
    {
      if (locID % (2 * phase) == 0)
        {
          int otherID = locID | phase;
          if (otherID < BLOCKSIZE)
            {
              d2 = maxD[otherID];
              if (d2 > maxD[locID])
                {
                  maxD[locID] = d2;
                  maxIdx[locID] = maxIdx[otherID];
                }
            }
        }
      __syncthreads ();
    }

  __syncthreads ();
  if (locID == 0)
    {
      locMax[blockIdx.x] = maxIdx[0];
      subDist[blockIdx.x] = maxD[0];
    }
}

//**********************************************
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  thrust::host_vector < float >h_x (N);
  thrust::host_vector < float >h_y (N);
  thrust::device_vector < float >d_x;
  thrust::device_vector < float >d_y;
  thrust::device_vector < float >d_distance (N);

  // data init.
  srand (time (0));
  for (int i = 0; i < N; i++)
    {
      h_x[i] = rand () % 1000;
      h_y[i] = rand () % 1000;
    }

  // Use Thrust to calculate distances
  d_x = h_x;
  d_y = h_y;
  thrust::transform (d_x.begin (), d_x.end (), d_y.begin (), d_distance.begin (), distFunct ());

  thrust::device_vector < float >::iterator max_dist = thrust::max_element (d_distance.begin (), d_distance.end ());
  float h_max = *max_dist;
  cout << "Max dist by GPU (Thrust): " << h_max << endl;

  // Switch to CUDA to find the index of the most distant point
  int grid = (N/2 + BLOCKSIZE - 1) / BLOCKSIZE;
  int *d_idx, *h_idx;
  float *h_largestDist, *d_largestDist;
  h_idx = new int[grid];
  h_largestDist = new float[grid];

  cudaMalloc ((void **) &d_idx, grid * sizeof (int));
  cudaMalloc ((void **) &d_largestDist, grid * sizeof (float));
  findMax <<< grid, BLOCKSIZE >>> (thrust::raw_pointer_cast (d_distance.data ()), N, d_idx, d_largestDist);
  cudaMemcpy (h_idx, d_idx, grid * sizeof (int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_largestDist, d_largestDist, grid * sizeof (float), cudaMemcpyDeviceToHost);

  // process now one result per block at the host side
  int best = 0;
  for (int i = 1; i < grid; i++)
    if (h_largestDist[i] > h_largestDist[best])
      best = i;
  cout << "Max dist by GPU: " << h_largestDist[best] << " at index " << h_idx[best] << endl;

  //*********************************
  // CPU-based calculation
  h_max = 0;
  int maxIdx = 0;
  for (int i = 0; i < N; i++)
    {
      float temp = h_x[i] * h_x[i] + h_y[i] * h_y[i];
      if (temp > h_max)
        {
          h_max = temp;
          maxIdx = i;
        }
    }
  cout << "Max dist by CPU: " << h_max << " at index " << maxIdx << endl;


  return 0;
}
