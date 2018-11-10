/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc quicksort_dynamic.cu -arch=compute_35 -code=sm_35 -rdc=true -o quicksort_dynamic
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>

using namespace std;

const int MAXRECURSIONDEPTH=16;

void dump (int *data, int N)
{
  for (int i = 0; i < N; i++)
    printf ("%i ", data[i]);
  printf ("\n");
}

//************************************************
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//************************************************
__device__ void swap (int *data, int x, int y)
{
  int temp = data[x];
  data[x] = data[y];
  data[y] = temp;
}

//************************************************
__device__ int partition (int *data, int N)
{
  int i = 0, j = N;
  int pivot = data[0];

  do
    {
      do
        {
          i++;
        }
      while (pivot > data[i] && i < N);

      do
        {
          j--;
        }
      while (pivot < data[j] && j > 0);
      swap (data, i, j);
    }
  while (i < j);
  // undo last swap
  swap (data, i, j);

  // fix the pivot element position
  swap (data, 0, j);
  return j;
}

//************************************************
__device__ void insertionSort (int *data, int N)
{
  int loc=1;
  while(loc < N)
  {
    int temp = data[loc];
    int i=loc-1;
    while(i>=0 && data[i] > temp)
    {
       data[i+1]=data[i];
       i--;
    }
    data[i+1] = temp;
    loc++;
  }
}
//************************************************
__global__ void QSort (int *data, int N, int depth)
{
  if(depth == MAXRECURSIONDEPTH)
  {
    insertionSort(data, N);  
    return ;
  }
  
  if (N <= 1)
    return;
  
  // break the data into a left and right part
  int pivotPos = partition (data, N);
  
  cudaStream_t s0, s1;
  // sort the left part if it exists
  if (pivotPos > 0)
    {
      cudaStreamCreateWithFlags (&s0, cudaStreamNonBlocking);
      QSort <<< 1, 1, 0, s0 >>> (data, pivotPos, depth+1);
      cudaStreamDestroy (s0);
    }

  // sort the right part if it exists
  if (pivotPos < N - 1)
    {
      cudaStreamCreateWithFlags (&s1, cudaStreamNonBlocking);
      QSort <<< 1, 1, 0, s1 >>> (&(data[pivotPos + 1]), N - pivotPos - 1, depth+1);
      cudaStreamDestroy (s1);
    }
}

//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc == 1)
    {
      fprintf (stderr, "%s N\n", argv[0]);
      exit (0);
    }
  int N = atoi (argv[1]);
  int *data;
  cudaMallocManaged ((void **) &data, N * sizeof (int));

  numberGen (N, 1000, data);

  QSort <<< 1, 1 >>> (data, N, 0);

  cudaDeviceSynchronize ();

  dump (data, N);

  // clean-up allocated memory
  cudaFree (data);
  return 0;
}
