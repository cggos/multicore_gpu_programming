/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp reentrant_test.cpp -o reentrant_test
 ============================================================================
 */
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <omp.h>
using namespace std;

#define MAXVALUE 10000
#define min(A,B) ((A<B) ? A : B)
#define max(A,B) ((A>B) ? A : B)
//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

int comp(const void *x, const void *y, void *thunk)
{
  int a = *(int *)x;
  int b = *(int *)y;
  return a-b;
}
//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc < 2)
    {
      cout << "Use : " << argv[0] << " numData\n";
      exit (1);
    }

  int N = atoi (argv[1]);
  int *data = new int[N];
  numberGen (N, MAXVALUE, data);

#pragma omp parallel num_threads(2)
  {
    qsort_r(data, N, sizeof(int), comp, NULL);
  }
  // Sanity check
  for (int i = 0; i < N - 1; i++)
    if (data[i + 1] < data[i])
      cout << "ERROR!\n";

  delete[]data;
  return 0;
}
