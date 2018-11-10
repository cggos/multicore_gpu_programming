/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc iterators_example.cu -o iterators_example
 ============================================================================
 */
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/replace.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

//*********************************************************
template < typename T> void print (char *s, thrust::host_vector < T > &v)
{
  cout << s ;
  thrust::copy (v.begin (), v.end (), ostream_iterator < T > (cout, " "));
  cout << endl;
}
//*********************************************************
struct evenFunct
{
  __host__ __device__
   bool operator()(int i)
   {
     return i%2==0;
   }
};
//*********************************************************
struct pivotFunct
{
  int pivot;
  pivotFunct(int p) : pivot(p){}
  
  __host__ __device__
   bool operator()(int i)
   {
     return i<pivot;
   }
};
//*********************************************************
int main ()
{
  int aux[] = { 5, 1, 3, 3, 2, 4, 2, 7, 6, 7 };
  char aux2[] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
  int numItems = sizeof(aux)/sizeof(int);
  thrust::host_vector < int >h_keys (aux, aux + numItems);
  thrust::host_vector < char >h_values(aux2, aux2 + numItems);

  thrust::host_vector<int> dest_keys(numItems);
  thrust::host_vector<char> dest_values(numItems);
  
  thrust::host_vector<int>::iterator newEnd = thrust::copy_if(h_keys.begin(), h_keys.end(), dest_keys.begin(), evenFunct());
  dest_keys.resize( newEnd - dest_keys.begin());
  print("copy_if : ", dest_keys);
  // Output is:
  // copy_if : 2 4 2 6 

  dest_keys.resize(numItems);
  newEnd = thrust::remove_copy(h_keys.begin(), h_keys.end(), dest_keys.begin(), 3);
  dest_keys.resize( newEnd - dest_keys.begin());
  print("remove_copy : ", dest_keys);
  // Output is:
  // remove_copy : 5 1 2 4 2 7 6 7 
  
  dest_keys.resize(numItems);
  newEnd = thrust::unique_copy(h_keys.begin(), h_keys.end(), dest_keys.begin());
  dest_keys.resize( newEnd - dest_keys.begin());
  print("unique_copy : ", dest_keys);
  // Output is:
  // unique_copy : 5 1 3 2 4 2 7 6 7 

  thrust::pair<thrust::host_vector<int>::iterator, thrust::host_vector<char>::iterator> endsPair = thrust::unique_by_key_copy(h_keys.begin(), h_keys.end(), h_values.begin(), dest_keys.begin(), dest_values.begin());
  dest_keys.resize(endsPair.first - dest_keys.begin());
  dest_values.resize(endsPair.second - dest_values.begin());
  print("unique_by_key_copy (keys)  : ", dest_keys);
  print("unique_by_key_copy (values): ", dest_values);
  // Output is:
  // unique_by_key_copy (keys)  : 5 1 3 2 4 2 7 6 7 
  // unique_by_key_copy (values): A B C E F G H I J 
  
  thrust::sort(h_keys.begin(), h_keys.end());
  dest_keys.resize(numItems);
  newEnd = thrust::unique_copy(h_keys.begin(), h_keys.end(), dest_keys.begin());
  dest_keys.resize( newEnd - dest_keys.begin());
  print("unique_copy for sorted : ", dest_keys);
  // Output is:
  // unique_copy for sorted : 1 2 3 4 5 6 7 

  thrust::replace_if(h_keys.begin(), h_keys.end(), evenFunct(), 0);
  print("replace_if : ", h_keys);
  // Output is:
  // replace_if : 1 0 0 3 3 0 5 0 7 7 

  thrust::partition(h_keys.begin(), h_keys.end(), pivotFunct( h_keys[0] ));
  print("partition : ", h_keys);
  // Output is:
  // partition : 0 0 0 0 1 3 3 5 7 7

  return 0;
}
