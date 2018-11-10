/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc max3d.cu -o max3d
 ============================================================================
 */
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/tuple.h>
#include <math.h>

using namespace std;

// Calculate the square of the distance
struct distSqrFunct
{
  template < typename Tuple >
  __host__ __device__ 
  float operator() (Tuple t)
  {
    int x = thrust::get < 0 > (t);
    int y = thrust::get < 1 > (t);
    int z = thrust::get < 2 > (t);
    return x * x + y * y + z * z;
  }
};

//****************************************************
struct maxFunct
{
  thrust::device_ptr < int >dis;
    maxFunct (thrust::device_ptr < int >d):dis (d)  {}

  __host__ __device__ 
  int operator() (int idx1, int idx2)
  {
    if (dis[idx1] > dis[idx2])
      return idx1;
    return idx2;
  }
};

//****************************************************
int main (int argc, char **argv)
{
  // initialize the RNG
  thrust::default_random_engine rng (time(0));
  thrust::uniform_int_distribution<int> uniDistr(-10000,10000);
  
  int N = atoi (argv[1]);

  // generate the data on the host and move them to the device
  thrust::device_vector < int >x (N);
  thrust::device_vector < int >y (N);
  thrust::device_vector < int >z (N);
  thrust::device_vector < int >dis (N);
  thrust::host_vector<int> aux(N);
  for (int i = 0; i < x.size (); i++) aux[i] = uniDistr(rng);
  x = aux;
  for (int i = 0; i < x.size (); i++) aux[i] = uniDistr(rng);
  y = aux;
  for (int i = 0; i < x.size (); i++) aux[i] = uniDistr(rng);
  z = aux;
  
  // "zip" together the 3 arrays into one
  typedef thrust::device_vector < int >::iterator DVIint;
  typedef thrust::tuple < DVIint, DVIint, DVIint > tTuple;
  tTuple a = thrust::make_tuple (x.begin (), y.begin (), z.begin ());
  tTuple b = thrust::make_tuple (x.end (), y.end (), z.end ());

  // calculate the distance for each point
  thrust::transform (thrust::make_zip_iterator (a), thrust::make_zip_iterator (b), dis.begin (), distSqrFunct ());

  // initialize the functor that will find the maximum distance, so that it has access to the distance data
  maxFunct f (dis.data());
  
  // reduce the index of the most distant point
  int furthest = thrust::reduce (thrust::counting_iterator < int >(0),
                             thrust::counting_iterator < int >(N),
                             0,
                             f);

  float maxDist = dis[furthest]; // get max distance^2 from the device memory
  cout << "The most distant point is the " << furthest << "-th one, with a distance of " << sqrt(maxDist) << endl;
  return 0;

}
