/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp fibo.cpp -o fibo
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int fib (int i)
{
  int t1, t2;
  if (i == 0 || i == 1)
    return 1;
  else
    {
#pragma omp task shared(t1) if(i>25) mergeable
      t1 = fib (i - 1);

// #pragma omp task shared(t2) if(i>25) mergeable
      t2 = fib (i - 2);
#pragma omp taskwait
      return t1 + t2;
    }
}

//---------------------------------------
int main (int argc, char *argv[])
{
  // build a sample list
  int N = atoi (argv[1]);

#pragma omp parallel
  {
#pragma omp single
    {
      cout << fib (N) << endl;
    }
  }

  return 0;
}
