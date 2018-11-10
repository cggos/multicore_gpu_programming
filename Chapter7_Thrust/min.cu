/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc min.cu -o min
 ============================================================================
 */
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include <stdlib.h>
#include <time.h>

using namespace std;
//***************************************************
struct functor
{
  __host__ __device__
  float operator()(const float &x) const{
    return x*x;    
  }
};
//***************************************************
int main(int argc, char **argv)
{ 
  float st, end;
  st = atof(argv[1]);
  end = atof(argv[2]);
  int dataPoints = atoi(argv[3]);
  float step= (end-st)/dataPoints;
  
  thrust::device_vector<float> d_x(dataPoints);
  thrust::device_vector<float> d_y(dataPoints);
  thrust::sequence(d_x.begin(), d_x.end(), st, step);  

  functor f; 
  thrust::transform(d_x.begin(), d_x.end(), d_y.begin(), f);

  int idx = thrust::min_element(d_y.begin(), d_y.end())  - d_y.begin();
  cout << "Function minimum over ["<< st <<"," << end << "] occurs at " << d_x[idx] << endl;

  return 0;   
}