/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc dna_fusion.cu -o dna_fusion
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

//*********************************************************
template < typename D > 
struct Pass1_funct
{
  thrust::device_ptr < char >S;
  char T;                       // T single character used here
  thrust::device_ptr< D >prev;

    Pass1_funct (thrust::device_ptr < char >s, char t, thrust::device_ptr < D >p):S (s), T (t), prev (p) {}

  // just finds the maximum that can be obtained from 
  // the two cells from the previous iteration
  __host__ __device__ D operator () (const D & j) const
  {
    D max = prev[j];
    // optimized check that avoids path divergence
    D tmp = prev[j - 1] + (S[j - 1] == T);

    if (max < tmp)  max = tmp;

    return max;
  }
};
//*********************************************************

int main ()
{
  char *S = "GAATTCAGTTA"; // sample data
  char *T = "GGATCGA";
  int N = strlen (S);
  int M = strlen (T);

  // allocate and initialize the equivalent of a (M+1)x(N+1) matrix
  thrust::device_vector < int >H[M + 1];
  thrust::device_vector < int >aux;
  for (int i = 0; i < M + 1; i++)
    H[i].resize (N + 1);

  thrust::fill (H[0].begin (), H[0].end (), 0);
  for (int j = 1; j < M + 1; j++)
    H[j][0] = 0;

  // transfer to the device the big DNA strand
  thrust::device_vector < char >d_S (N);
  thrust::copy (S, S + N, d_S.begin ());

  thrust::counting_iterator < int >c (1);

  // fill-in the DP table, row-by-row
  for (int j = 1; j < M + 1; j++)
    {
      char oneOfT = T[j - 1];
      thrust::transform_inclusive_scan (c, c + N,
                                        H[j].begin () + 1,
                                        Pass1_funct < int >(d_S.data (), oneOfT, H[j - 1].data ()),
                                        thrust::maximum < int >());
    }

  // output the matrix 
//   for (int j = 0; j < M + 1; j++)      
//   {
//     for (int i = 0; i < N + 1; i++)
//       cout << H[j][i] << " ";
//     cout << "\n";
//   }

  cout << "Best matching score is " << H[M][N]  << endl;
  return 0;
}
