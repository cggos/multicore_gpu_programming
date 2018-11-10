/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc commExample.c -o commExample
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>

int main (int argc, char **argv)
{
  int num, i, rank, localRank;
  MPI_Group all, odd, even;
  MPI_Comm oddComm, evenComm;
  char mess[11];

  MPI_Init (&argc, &argv);
  // copy all the processes in group "all"
  MPI_Comm_group (MPI_COMM_WORLD, &all);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  int grN = 0;
  int ranks[num / 2];

  for (i = 0; i < num; i += 2)
    ranks[grN++] = i;

  // extract from "all" only the odd ones
  MPI_Group_excl (all, grN, ranks, &odd);
  // sutract odd group from all to get the even ones
  MPI_Group_difference (all, odd, &even);

  MPI_Comm_create (MPI_COMM_WORLD, odd, &oddComm);
  MPI_Comm_create (MPI_COMM_WORLD, even, &evenComm);
  
  // check group membership
  MPI_Group_rank (odd, &localRank);
  if (localRank != MPI_UNDEFINED)
    {
      if (localRank == 0)       // local group root, sets-up message
        strcpy (mess, "ODD GROUP");
      MPI_Bcast (mess, 11, MPI_CHAR, 0, oddComm);
      MPI_Comm_free (&oddComm);  // free communicator in processes where it is valid
    }
  else
    {
      MPI_Comm_rank (evenComm, &localRank);
      if (localRank == 0)       // local group root, sets-up message
        strcpy (mess, "EVEN GROUP");
      MPI_Bcast (mess, 11, MPI_CHAR, 0, evenComm);
      MPI_Comm_free (&evenComm);
    }

  printf ("Process %i with local rank %i received %s\n", rank, localRank, mess);

  // free up memory
  MPI_Group_free (&all);
  MPI_Group_free (&odd);
  MPI_Group_free (&even);
  MPI_Finalize ();
  return 0;
}
