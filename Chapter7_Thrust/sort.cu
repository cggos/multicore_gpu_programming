/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc sort.cu -o sort
 ============================================================================
 */
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void print(char c, thrust::host_vector<int> &v)
{
   cout << "Version " << c << " : ";  
   for(int i=0;i<v.size(); i++)
       cout << v[i] << " ";
   cout << endl;
}

void printD(char c, thrust::device_vector<int> &v)
{
   cout << "Version " << c << " : ";  
   for(int i=0;i<v.size(); i++)
       cout << v[i] << " ";
   cout << endl;
}

__global__ void kernFoo(int *x, int N)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%i %i \n",ID, x[ID]);
  x[ID] = ID*2;
}

int main ()
{
  srand(time(0));
  
  thrust::host_vector<int> h(100);
  thrust::host_vector<int> k(100);
  for(int i=0;i<h.size(); i++)
  {   h[i] = rand() % 1000;
  k[i] = 100 -i;
  }
  
  thrust::sort(h.begin(), h.end());
  print('A', h);

  thrust::sort_by_key(k.begin(), k.end(), h.begin());
  print('B', h);

  return 0;  
}
