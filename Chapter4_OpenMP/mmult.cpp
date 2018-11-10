/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp mmult.cpp -o mmult
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main (int argc, char **argv)
{
  int K = 10, L = 20, M = 5;
  double A[K][L];
  double B[L][M];
  double C[K][M];

#pragma omp parallel for collapse(3)
  for (int i = 0; i < K; i++)
    for (int j = 0; j < M; j++)
      for (int k = 0; k < L; k++)
        C[i][j] += A[i][k] * B[k][j];


  return 0;
}
