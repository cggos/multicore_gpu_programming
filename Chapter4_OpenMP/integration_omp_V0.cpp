/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Wrong OpenMP integration of testf
 To build use  : g++ -fopenmp integration_omp_V0.cpp -o integration_omp_V0
 ============================================================================
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

//---------------------------------------
double testf (double x)
{
  return x * x + 2 * sin (x);
}

//---------------------------------------
double integrate (double st, double en, int div, double (*f) (double))
{
  double localRes = 0;
  double step = (en - st) / div;
  double x;
  x = st;
  localRes = f (st) + f (en);
  localRes /= 2;
  for (int i = 1; i < div; i++)
    {
      x += step;
      localRes += f (x);
    }
  localRes *= step;

  return localRes;
}

//---------------------------------------
int main (int argc, char *argv[])
{

  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " start end divisions\n";
      exit (1);
    }
  double start, end;
  int divisions;
  start = atof (argv[1]);
  end = atof (argv[2]);
  divisions = atoi (argv[3]);

  // get the number of threads for next parallel region
  int N = omp_get_max_threads ();
  divisions = (divisions / N) * N;    // make sure divisions is a multiple of N
  double step = (end - start) / divisions;

  double finalRes = 0;
#pragma omp parallel
  {
    int localDiv = divisions / N;
    int ID = omp_get_thread_num ();
    double localStart = start + ID * localDiv * step;
    double localEnd = localStart + localDiv * step;
    finalRes += integrate (localStart, localEnd, localDiv, testf);
  }

  cout << finalRes << endl;

  return 0;
}
