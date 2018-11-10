/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc sum.cu -o sum
 ============================================================================
 */
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace std;

int main (int argc, char **argv)
{
  srand (time (NULL));
  int N = atoi (argv[1]);
  thrust::host_vector < int >h_d (N);
  for (int i = 0; i < N; i++)
    h_d[i] = rand () % 10000;


  thrust::device_vector < int >d_d (N);
  d_d = h_d;

  cout << "Average computed on CPU :" << thrust::reduce (h_d.begin (), h_d.end ()) * 1.0 / N << endl;

  cout << "Average computed on GPU :" << thrust::reduce (d_d.begin (), d_d.end ()) * 1.0 / N << endl;

  return 0;
}
