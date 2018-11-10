/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC ping_pong.cpp -o ping_pong
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

using namespace std;

#define MESG_TAG 0
#define END_TAG 1
#define MAX_MESG 1000000

const int REP=10;

int main (int argc, char *argv[])
{
  int size, rank;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int mesg_size;
  int tag;
  char *buffer;
  double start_time, end_time;
  MPI_Status status;

  buffer = new char[MAX_MESG];

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name (processor_name, &namelen);

  printf ("Process %d of %d on %s\n", rank, size, processor_name);
  if (size < 2)
    {
      printf ("Need more than 1 processor to run\n");
      exit (1);
    }

  if (rank == 0)
    {
      for(int mesg_size = 0; mesg_size<=MAX_MESG; mesg_size += 1000)
      {
	      start_time = MPI_Wtime ();
	      for(int i=0;i<REP;i++)
	      {
	      MPI_Send (buffer, mesg_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
	      MPI_Recv (buffer, MAX_MESG, MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	      }
	      
	      end_time = MPI_Wtime ();
	printf("%i %lf\n", mesg_size, (end_time - start_time)/2/REP);
      }
      
      tag = END_TAG;
      MPI_Send (buffer, 0, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
    }
  else
    {
      while (1)
	{
	  MPI_Recv (buffer, MAX_MESG, MPI_CHAR, 0, MPI_ANY_TAG,
		    MPI_COMM_WORLD, &status);
	  if (status.MPI_TAG == END_TAG)
	    break;
	  MPI_Get_count (&status, MPI_CHAR, &mesg_size);
	  MPI_Send (buffer, mesg_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
	}
    }

  delete []buffer;
  MPI_Finalize ();
  return 0;
}
