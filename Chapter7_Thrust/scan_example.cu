/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc scan_example.cu -o scan_example
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
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

int main ()
{
  int data[] = { 10, 1, 34, 7, 8, 10, 17 };
  int numItems = sizeof(data)/sizeof(int);
  thrust::host_vector < int >h_data (data, data + numItems);
  thrust::host_vector < int >h_r;

  thrust::device_vector < int >d_data(h_data);
  thrust::device_vector < int >d_r(numItems);
 
  thrust::inclusive_scan(d_data.begin (), d_data.end (), d_r.begin());
  h_r = d_r;
  print("Inclusive scan : ", h_r);
  // Output is:
  // Inclusive scan : 10 11 45 52 60 70 87 
  
  thrust::exclusive_scan(d_data.begin (), d_data.end (), d_r.begin());
  h_r = d_r;
  print("Exclusive scan : ", h_r);
  // Output is:
  // Exclusive scan : 0 10 11 45 52 60 70

  thrust::inclusive_scan(d_data.begin (), d_data.end (), d_r.begin(), thrust::multiplies<int>());
  h_r = d_r;
  print("Inclusive scan product : ", h_r);
  // Output is:
  // Inclusive scan product : 10 10 340 2380 19040 190400 3236800
  
  return 0;
}
