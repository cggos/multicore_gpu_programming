/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Parallel matrix mult. using OpenMP
 To build use  : g++ -fopenmp matrixMult_pri.cpp -o matrixMult_pri
 ============================================================================
 */

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <omp.h>
using namespace std;

#define MAXVALUE 100
#define min(A,B) ((A<B) ? A : B)
#define max(A,B) ((A>B) ? A : B)
//------------------------------------
void numberGen (int N, int max, double *store)
{
  int i;
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//--------------------------------------------------------
void mmult (double *A, double *B, double *C, int N, int K, int M)
{
  // implied schedule(static, N*M / numThreads)
#pragma omp parallel for collapse(2)  
  for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
      {
        double temp = 0;
        for (int l = 0; l < K; l++)
          temp += A[i * K + l] * B[l * M + j];
	C[i * M + j] = temp;
      }
}

//--------------------------------------------------------
void dump (double *x, int N, int M)
{
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
        cout << x[i * M + j] << " ";
      cout << endl;
    }
  cout << "----------------------------------------\n";
}

//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc != 4)
    {
      cout << "Use : " << argv[0] << " N K M\n";
      exit (1);
    }

  srand (time (0));
  int N = atoi (argv[1]);
  int K = atoi (argv[2]);
  int M = atoi (argv[3]);
  double *A = new double[N * K];
  double *B = new double[K * M];
  double *C = new double[N * M];

  numberGen (N * K, MAXVALUE, A);
  numberGen (K * M, MAXVALUE, B);

  double t = omp_get_wtime ();
  mmult (A, B, C, N, K, M);
  cout << omp_get_wtime () - t << endl;

//   dump (A, N, K);
//   dump (B, K, M);
//   dump (C, N, M);


  delete[]A;
  delete[]B;
  delete[]C;
  return 0;
}
