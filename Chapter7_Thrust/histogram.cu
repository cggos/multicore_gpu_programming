/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc histogram.cu -o histogram
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
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
template < typename T > void histogram (thrust::device_vector < T > &data, thrust::device_vector < int >&hist)
{
  thrust::sort (data.begin (), data.end ());
  T min = data[0];
  T max = data[data.size () - 1];
  T range = max - min + 1;
  hist.resize (range);

  thrust::device_vector < int >aux;
  aux.push_back (0);
  aux.resize (hist.size () + 1);

  thrust::counting_iterator < int >search (min);
  thrust::upper_bound (data.begin (), data.end (), search, search + range, aux.begin () + 1);

  thrust::transform (aux.begin () + 1, aux.end (), aux.begin (), hist.begin (), thrust::minus < T > ());
}

//*********************************************************
int main (int argc, char **argv)
{
  int N = atoi (argv[1]);
  thrust::host_vector < int >h_x (N);
  thrust::host_vector < int >h_hist;
  thrust::device_vector < int >d_x;
  thrust::device_vector < int >d_hist;

  srand (time (0));
  for (int i = 0; i < N; i++)
    h_x[i] = rand () % 20;

  d_x = h_x;

  histogram (d_x, d_hist);
  h_hist = d_hist;
  print ("Hist ", h_hist);

  return 0;
}
