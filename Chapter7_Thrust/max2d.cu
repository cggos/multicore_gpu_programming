/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc max2d.cu -o max2d
 ============================================================================
 */
// Finding the distance of the most distant 2D point
// Jan. 2014, G. Barlas

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <stdlib.h>
#include <thrust/extrema.h>

using namespace std;
struct distFunct
{
  __host__ __device__
  float operator()(float &x, float &y)
  {
    return x*x + y*y;
  }
};

int main(int argc, char **argv)
{
  int N = atoi(argv[1]);
  thrust::host_vector<float> h_x(N);
  thrust::host_vector<float> h_y(N);
  thrust::device_vector<float> d_x;
  thrust::device_vector<float> d_y;
  thrust::device_vector<float> d_distance(N);
  
  srand(time(0));
  for(int i=0;i<N;i++)
  {
    h_x[i] = rand()%1000;
    h_y[i] = rand()%1000;
  }
  
  d_x= h_x;
  d_y= h_y;
  thrust::transform(d_x.begin(), d_x.end(),
                    d_y.begin(),
		    d_distance.begin(), distFunct() );
// OR		    
//   distFunct f();
//   thrust::transform(d_x.begin(), d_x.end(),
//                     d_y.begin(),
// 		    d_distance.begin(), f);
  thrust::device_vector<float>::iterator max_dist = thrust::max_element(d_distance.begin(), d_distance.end());
  float h_max = *max_dist;
  cout << "Max dist by GPU: " << h_max << endl; 
  

  h_max = 0;
  for(int i=0;i<N;i++)
  {
    float temp = h_x[i]* h_x[i] + h_y[i]* h_y[i];
    if(temp>h_max) h_max = temp;
  }
  cout << "Max dist by CPU: " << h_max << endl; 
  
 return 0; 
}