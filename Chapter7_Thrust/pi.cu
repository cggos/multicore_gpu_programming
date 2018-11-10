/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc pi.cu -o pi
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/random.h>
#include <thrust/functional.h>

#include <iostream>
#include <iomanip>

using namespace std;

//*********************************************************
struct MonteCarloPi
{
  int seed;
  int pointsPerThread;

  MonteCarloPi(int s, int p) : seed(s), pointsPerThread(p){}
  
  __host__ __device__
  long operator()(int segment)
  {
    double x, y, distance;
    long inside=0;
    thrust::default_random_engine rng(seed);
    rng.discard(segment * 2 * pointsPerThread);
    
    thrust::uniform_real_distribution<double> uniDistr(0,1);
    
    for(int i=0;i<pointsPerThread;i++)
    {
        x = uniDistr(rng);    // generate an x and y coordinates in the
        y = uniDistr(rng);    // [0,1]x[0,1] part of the plane
        distance = x*x + y*y;
        inside += ( distance <= 1 ) ; // optimized circle check 
    }
    return inside;
  }
};

//*********************************************************
int main (int argc, char **argv)
{

  int N = atoi (argv[1]);
  int M = atoi (argv[2]);

  N = (N+M-1)/M * M; // make sure N is a multiple of M
  
  long total = thrust::transform_reduce(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N/M), MonteCarloPi(0, M), 0, thrust::plus<int>());
  cout << setprecision(15);
  cout << 1.0L * total / N * 4.0L << endl;
  return 0;
}
