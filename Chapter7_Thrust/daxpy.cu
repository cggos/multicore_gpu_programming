/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc daxpy.cu -o daxpy
 ============================================================================
 */
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>

using namespace std;
//************************************************************
template < typename t > void print (char c, thrust::host_vector < t > &v)
{
  cout << "Version " << c << " : ";
  thrust::copy (v.begin (), v.end (), ostream_iterator < t > (cout, ", "));
  cout << endl;
}

//************************************************************
struct saxpy
{
  double a;
  saxpy ():a (1.0)  {};
  saxpy (double i):a (i)  {};

  __host__ __device__ 
  double operator () (double &x, double &y)  {
    return a * x + y;
  }
};

//************************************************************

int main ()
{
  thrust::device_vector < double >d_x (100);
  thrust::device_vector < double >d_y (100);
  thrust::device_vector < double >d_res (100);
  thrust::sequence (d_x.begin (), d_x.end (), 0.0, .1);
  thrust::fill (d_y.begin (), d_y.end (), 0.5);

  saxpy funct;
  funct.a = 1.2;

  thrust::transform (d_x.begin (), d_x.end (), d_y.begin (), d_res.begin (), funct);
//** alternative call using an unnamed functor
// thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_res.begin(), saxpy(1.2)); 

  thrust::host_vector < double >h_res (d_res);
  print < double >('A', h_res);

  return 0;
}
