/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp nested_flowDepFixed.cpp -o nested_flowDepFixed
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main (int argc, char **argv)
{
  int N = atoi (argv[1]), M = atoi (argv[2]);
  double **data = new double *[N];
  for (int i = 0; i < N; i++)
    data[i] = new double[M];

  // init with sample values
  for (int i = 0; i < N; i++)
    data[i][0] = 1;

  for (int j = 0; j < M; j++)
    data[0][j] = 1;

  int smallDim;
  int largeDim;
  if (N > M)
    {
      smallDim = M;
      largeDim = N;
    }
  else
    {
      smallDim = N;
      largeDim = M;
    }

  // compute
  for (int diag = 1; diag <= N + M - 3; diag++)
    {
      int diagLength = diag;
      if (diag + 1 >= smallDim)
        diagLength = smallDim - 1;
      if (diag + 1 >= largeDim)
        diagLength = (smallDim - 1) - (diag - largeDim) - 1;
//        cout << "D : " << diagLength << "     ";
#pragma omp parallel for default(none) shared(data, diag, diagLength, N, M)
      for (int k = 0; k < diagLength; k++)
        {
          int i = diag - k;
          int j = k + 1;
          if (diag > N - 1)
            {
              i = N - 1 - k;
              j = diag - (N - 1) + k + 1;
            }
//            cout << "(" << i << "," << j << ") ";
          data[i][j] = data[i - 1][j] + data[i][j - 1] + data[i - 1][j - 1];
        }
//        cout << endl;
    }
  //output
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
        cout << data[i][j] << " ";
      cout << endl;
    }

  return 0;
}
