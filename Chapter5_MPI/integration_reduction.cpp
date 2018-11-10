/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : Dec. 2014, Nov. 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC integration_reduction.cpp -o integration_reduction
 ============================================================================
 */
#include <mpi.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

//---------------------------------------
double testf (double x) {
  return x * x + 2 * sin (x);
}

//---------------------------------------
//calculate and return area  
double integrate (double st, double en, int div, double (*f) (double)) {
  double localRes = 0;
  double step = (en - st) / div;
  double x;
  x = st;
  localRes = f (st) + f (en);
  localRes /= 2;
  for (int i = 1; i < div; i++) {
      x += step;
      localRes += f (x);
    }
  localRes *= step;

  return localRes;
}

//---------------------------------------
int main (int argc, char *argv[]) {

  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;

  if (argc == 1) {
      if (rank == 0)
        cerr << "Usage " << argv[0] << " start end divisions\n";
      exit (1);
    }
  double start, end;
  int divisions;
  start = atof (argv[1]);
  end = atof (argv[2]);
  divisions = atoi (argv[3]);

  double locSt, locEnd, rangePerProc;
  int locDiv;
  locDiv = ceil (1.0 * divisions / N);
  rangePerProc = (end - start) / N;
  locSt = start + rangePerProc * rank;
  locEnd = (rank == N - 1) ? end : start + rangePerProc * (rank + 1);
  double partialResult = integrate (locSt, locEnd, locDiv, testf);
  double finalRes;
  MPI_Reduce (&partialResult, &finalRes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
    cout << finalRes << endl;
  MPI_Finalize ();
  return 0;
}
