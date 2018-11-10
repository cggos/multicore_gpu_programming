/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc histogram_sparse.cu -o histogram_sparse
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

//*********************************************************
template < typename T > void print (char *s, thrust::host_vector < T > &v)
{
  cout << s << ":";
  thrust::copy (v.begin (), v.end (), ostream_iterator < T > (cout, " "));
  cout << endl;
}

//*********************************************************
template < typename T >
void histogram_sparse (thrust::device_vector < T > &data, 
                       thrust::device_vector < T > &value,
                       thrust::device_vector < int >&count)
{
  thrust::sort (data.begin (), data.end ());

  // calculate how many different values exist in the vector
  // by comparing successive values in the sorted data.
  // For every different pair of keys (i.e. a change from one set to the next)
  // a value of 1 is produced and summed up 
  int numBins = thrust::inner_product (data.begin (), data.end () - 1,
                                       data.begin () + 1,
                                       0,
                                       thrust::plus < int >(),
                                       thrust::not_equal_to < T > ());

  // output vectors are resized to fit the results
  value.resize (numBins);
  count.resize (numBins);
  
  // the groups of identical keys, get their values (1) summed up
  // producing as a result a count 
  thrust::reduce_by_key (data.begin (), data.end (), 
                         thrust::constant_iterator < int >(1),
                         value.begin (),
                         count.begin ());
}

//*********************************************************
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  thrust::host_vector < int >h_x (N);
  thrust::host_vector < int >h_value;
  thrust::host_vector < int >h_count;
  thrust::device_vector < int >d_x;
  thrust::device_vector < int >d_value;
  thrust::device_vector < int >d_count;

  srand (time (0));
  for (int i = 0; i < N; i++)
    h_x[i] = rand () % 10000;

  d_x = h_x;

  histogram_sparse (d_x, d_value, d_count);
  h_value = d_value;
  h_count = d_count;
  print ("Values ", h_value);
  print ("Counts ", h_count);

  return 0;
}
