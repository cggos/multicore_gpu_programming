/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc perm_iter.cu -o perm_iter
 ============================================================================
 */
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/permutation_iterator.h>
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




int main ()
{
  srand(time(0));
  
  thrust::device_vector<int> d(100);
  thrust::device_vector<int> map(10);
  for(int i=0;i<d.size(); i++)
      d[i] = rand() % 1000;
  
  for(int i=0;i<10; i++)
      map[i] = 10*i;
  
  
  int sum = thrust::reduce( thrust::make_permutation_iterator(d.begin(), map.begin()),
                 thrust::make_permutation_iterator(d.begin(), map.end()) );
  printf("%i\n",sum);
  for(int i=0;i<10; i++)
      cout << d[i] << " ";
  
  return 0;  
}
