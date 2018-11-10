/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc buffered_example.c -o buffered_example
 ============================================================================
 */
#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define COMMBUFFSIZE 1024	/* This would be too small under most circumstances */
#define MAXMSGSIZE 10
#define MSGTAG 0

int main (int argc, char **argv)
{
  int rank, num, i;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num);

  if (rank == 0)
    {
      // allocate buffer space and designate it for MPI use
      unsigned char *buff =
	(unsigned char *) malloc (sizeof (unsigned char) * COMMBUFFSIZE);
      MPI_Buffer_attach (buff, COMMBUFFSIZE);
      char *msg = "Test msg";
      for (i = 1; i < num; i++)
	MPI_Bsend (msg, strlen (msg) + 1, MPI_CHAR, i, MSGTAG,
		   MPI_COMM_WORLD);
      // detach and release buffer space
      unsigned char *bptr;
      int bsize;
      MPI_Buffer_detach (&bptr, &bsize);
      free (bptr);

    }
  else
    {
      MPI_Status status;
      char msg[MAXMSGSIZE];
      MPI_Recv (msg, MAXMSGSIZE, MPI_CHAR, 0, MSGTAG, MPI_COMM_WORLD, &status);	// no change on receiving end    
      printf ("%s\n", msg);
    }

  MPI_Finalize ();
  return 0;
}
