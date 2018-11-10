/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc errorHandling.c -o errorHandling
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>
#define MESSTAG 0
#define MAXLEN 100

void customErrHandler(MPI_Comm *comm, int *errcode, ...)
{
  printf("Error %i\n", *errcode);
}

int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);
  MPI_Errhandler eh;
  
  MPI_Comm_create_errhandler(customErrHandler, &eh);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, eh);
  MPI_Comm c;
  
  int rank, num, i;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  if (rank == 0)
    {
      char mess[] = "Hello World";
      for (i = 1; i < num; i++)
	MPI_Send (mess, strlen (mess) + 1, MPI_CHAR, i, MESSTAG, c);
    }
  else
    {
      char mess[MAXLEN];
      MPI_Status status;
      MPI_Recv (mess, MAXLEN, MPI_DOUBLE, 0, MESSTAG, MPI_COMM_WORLD, &status);
      printf ("%i received %s\n", rank, mess);
    }
  MPI_Finalize ();
  return 0;
}
