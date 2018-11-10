/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc dna.cu -o dna
 ============================================================================
 */
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <iostream>

using namespace std;

struct Phase1_funct
{
  thrust::device_ptr < char >S;
  char T; // T single character used here
  thrust::device_ptr < int >prev;

  Phase1_funct (thrust::device_ptr < char >s, char t, thrust::device_ptr < int >p):S (s), T (t), prev (p) {}

  // just finds the maximum that can be obtained from 
  // the two cells from the previous iteration
  __host__ __device__ int operator () (int &j) const
  {
    int max = prev[j];

    int tmp = prev[j - 1];

    if (S[j - 1] == T)
      tmp++;
    if (max < tmp)
        max = tmp;

      return max;
  }
};
//*********************************************************

int main ()
{
  char *S = "GAATTCAGTTA";      // sample data
  char *T = "GGATCGA";
  int N = strlen (S);
  int M = strlen (T);

  // allocate and initialize the equivalent of 2 N+1-length vectors
  // [0] is used to hold at the end of each iteration, the last computed
  // row of the matrix
  thrust::device_vector < int >H[2];
  H[0].resize (N + 1);
  H[1].resize (N + 1);
  thrust::fill (H[0].begin (), H[0].end (), 0);

  // transfer the big DNA strand to the device 
  thrust::device_vector < char >d_S (N);
  thrust::copy (S, S + N, d_S.begin ());

  thrust::counting_iterator < int >c0 (1);
  thrust::counting_iterator < int >c1 (N + 1);

  for (int j = 0; j < M; j++)
    {
      char oneOfT = T[j];
      // first phase using the previous row in the matrix
      thrust::transform (c0, c1, H[1].begin () + 1, Phase1_funct (d_S.data (), oneOfT, H[0].data ()));

      // second phase using the current row in the matrix
      thrust::inclusive_scan (H[1].begin () + 1, H[1].end (), H[0].begin () + 1, thrust::maximum < int >());
    }

  cout << "Best matching score is " << H[0][N] << endl;


  // proof of concept CPU code
//   int dp[M+1][N+1];
//   for(int i=0;i<M+1;i++)
//     for(int j=0;j<N+1;j++)
//       dp[i][j] = 0;
//  
//   for(int i=1;i<M+1;i++)
//     for(int j=1;j<N+1;j++)
//     {
//       int tmp, max;
// 
//       max = dp[i-1][j];
// 
//       tmp = dp[i][j-1];
//       if(max < tmp)
//       max = tmp;
// 
//       tmp = dp[i-1][j-1];
//       if(S[j-1]==T[i-1])
//         tmp++;
//       if(max < tmp)
//       max = tmp;
//       
//        dp[i][j] = max;
//     }
// 
//  for(int i=0;i<M+1;i++)
//   {
//     for(int j=0;j<N+1;j++)
//        cout << dp[i][j] << " ";
//   cout << endl;
//   }

  return 0;
}
