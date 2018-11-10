/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc master.c -o master
 ============================================================================
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define MAXLEN 100
char buff[MAXLEN];

int main (int argc, char **argv)
{
  MPI_Status st;
  int procNum;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &procNum);
  while (--procNum)
    {
      MPI_Recv (buff, MAXLEN, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG,
		MPI_COMM_WORLD, &st);
      int aux;
      MPI_Get_count (&st, MPI_CHAR, &aux);
      buff[aux] = 0;
      printf ("%s\n", buff);
    }
  MPI_Finalize ();
}
