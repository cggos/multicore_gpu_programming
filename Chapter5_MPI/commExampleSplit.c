/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc commExampleSplit.c -o commExampleSplit
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>

int main (int argc, char **argv)
{
  int num, i, rank, localRank;
  MPI_Comm newComm;
  char mess[11];

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  MPI_Comm_split(MPI_COMM_WORLD, rank%2, rank/2, &newComm);
  
  if (rank == 0)       // root of even group
      strcpy (mess, "EVEN GROUP");
  else if(rank == 1)   // root of odd group
      strcpy (mess, "ODD GROUP");

  
  MPI_Bcast (mess, 11, MPI_CHAR, 0, newComm);
  MPI_Comm_rank (newComm, &localRank);
  MPI_Comm_free (&newComm);  // free communicator in processes where it is valid

  printf ("Process %i with local rank %i received %s\n", rank, localRank, mess);

  MPI_Finalize ();
  return 0;
}
