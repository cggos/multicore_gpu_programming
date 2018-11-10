/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc groupExample.c -o groupExample
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>

int main (int argc, char **argv)
{
  int num, i, rank;
  MPI_Group all, odd, even;

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

  // print group sizes
  if (rank == 0)
    {
      MPI_Group_size (odd, &i);
      printf ("Odd group has %i processes\n", i);
      MPI_Group_size (even, &i);
      printf ("Even group has %i processes\n", i);
    }

  // check group membership
  MPI_Group_rank (odd, &i);
  if (i == MPI_UNDEFINED)
    printf ("Process %i belongs to even group\n", rank);
  else
    printf ("Process %i belongs to odd group\n", rank);

  // free up memory
  MPI_Group_free (&all);
  MPI_Group_free (&odd);
  MPI_Group_free (&even);
  MPI_Finalize ();
  return 0;
}
