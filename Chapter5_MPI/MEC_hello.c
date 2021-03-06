/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc MEC_hello.c -o MEC_hello
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>
#define MESSTAG 0
#define MAXLEN 100

#define MEC(call) {int res; \
                   res = call; \
                   if(res != MPI_SUCCESS) { \
		        fprintf(stderr, "Call "#call" return error code %i\n", res); \
		        MPI_Abort(MPI_COMM_WORLD, res);}  }

int foo()
{
  return 1;
}

int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  MEC( foo());
  int rank, num, i;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  if (rank == 0)
    {
      char mess[] = "Hello World";
      for (i = 1; i < num; i++)
	MEC(MPI_Send (mess, strlen (mess) + 1, MPI_CHAR, 2*i, MESSTAG,MPI_COMM_WORLD));
    }
  else
    {
      char mess[MAXLEN];
      MPI_Status status;
      MEC(MPI_Recv (mess, MAXLEN, MPI_CHAR, 0, MESSTAG, MPI_COMM_WORLD, &status));
      printf ("%i received %s\n", rank, mess);
    }
  MPI_Finalize ();
  return 0;
}
