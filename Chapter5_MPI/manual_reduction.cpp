/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC manual_reduction.cpp -o manual_reduction
 ============================================================================
 */
//Reduction implementation
#include<mpi.h>
#include<iostream>

using namespace std;

//*****************************************
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;

  int partialSum = rank;
  int bitMask = 1;
  bool doneFlag = false;
  while (bitMask < N && !doneFlag)
    {
      int otherPartyID;
      if ((rank & bitMask) == 0)
        {
          otherPartyID = rank | bitMask;
          if (otherPartyID >= N)
            {
              bitMask <<= 1;
              continue;
            }
          int temp;
          MPI_Recv (&temp, 1, MPI_INT, otherPartyID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          partialSum += temp;
        }
      else
        {
          otherPartyID = rank ^ bitMask;
          doneFlag = true;
          MPI_Send (&partialSum, 1, MPI_INT, otherPartyID, 0, MPI_COMM_WORLD);
        }
      bitMask <<= 1;
    }

  if (rank == 0)
    cout << partialSum << endl;

  MPI_Finalize ();
  return 0;
}
