/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Example of MPI_Allgather
 To build use  : mpiCC allgatherMPI.cpp -o allgatherMPI
 ============================================================================
 */

#include<mpi.h>
#include<iostream>

const int K = 10;

using namespace std;

//*****************************************
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;


  double *localPart = new double[K];
  double *allParts = new double[K * N];

  for (int i = 0; i < K; i++)
    localPart[i] = rank;

  MPI_Allgather(localPart, K, MPI_DOUBLE, allParts, K, MPI_DOUBLE, MPI_COMM_WORLD);

  if (rank == 0)
    {
      for (int i = 0; i < K * N; i++)
        cout << allParts[i] << " ";
      cout << endl;
    }


  MPI_Finalize ();
  return 0;
}
