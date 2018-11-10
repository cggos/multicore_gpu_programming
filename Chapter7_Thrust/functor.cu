/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc functor.cu -o functor
 ============================================================================
 */
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <cuda.h>
#include <stdio.h>

using namespace std;

void print(char c, thrust::host_vector<double> &v)
{
   cout << "Version " << c << " : ";  
   for(int i=0;i<v.size(); i++)
       cout << v[i] << " ";
   cout << endl;
}
//************************************************************
struct saxpy
{
  double a;
  saxpy() : a(1.0) {};
  saxpy(double i) : a(i) {};
  
  __host__ __device__ double operator() ( double &x, double &y)
  {
    return a*x+y;
  }
};
//************************************************************
struct sqr
{
  __host__ __device__ double operator() ( double &x)
  {
    return x*x;
  }
};
//************************************************************

int main ()
{
  thrust::device_vector<double> d_x(100);
  thrust::device_vector<double> d_y(100);
  thrust::device_vector<double> d_res(100);
  thrust::sequence(d_x.begin(), d_x.end(), 0.0, .1);
  thrust::fill(d_y.begin(), d_y.end(), 0.5);
  
  saxpy funct;
  funct.a = 1.2;
  
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_res.begin(), funct); 
//   thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), d_res.begin(), saxpy(1.2)); 
  thrust::host_vector<double> h_res(d_res);
  print('A', h_res);

  double sum = thrust::reduce(d_res.begin(), d_res.end(), 0.0, thrust::plus<double>());
  
  double sum2 = thrust::transform_reduce(d_res.begin(), d_res.end(), sqr(), 0.0, thrust::plus<double>());
  printf("%lf %lf\n", sum, sum2);
  return 0;  
}
