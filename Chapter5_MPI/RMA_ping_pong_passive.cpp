/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC RMA_ping_pong_passive.cpp -o RMA_ping_pong_passive
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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

  if (size < 2)
    {
      printf ("Need more than 1 processor to run\n");
      exit (1);
    }

  MPI_Win win;
  int *ptr;
  ptr = (int *)buffer;
  MPI_Win_create(buffer, MAX_MESG, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  
  if (rank == 0)
    {
      for(int mesg_size = 0; mesg_size<=MAX_MESG; mesg_size += 1000)
      {
	      start_time = MPI_Wtime ();
	      
	      for(int i=0;i<REP;i++)
	      {
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE,1,0, win);
		*ptr = mesg_size;
                MPI_Put(buffer, mesg_size, MPI_UNSIGNED_CHAR, 1, 0, MAX_MESG, MPI_UNSIGNED_CHAR, win);
		MPI_Win_unlock(1, win);
	      }
	      
	      end_time = MPI_Wtime ();
	printf("%i %lf\n", mesg_size, (end_time - start_time)/REP);
      }

      
    }
//   else
//     {
//       for(int mesg_size = 0; mesg_size<=MAX_MESG; mesg_size += 1000)
//       {
// 	      for(int i=0;i<REP;i++)
// 	      {
// 		MPI_Win_lock(MPI_LOCK_EXCLUSIVE,1,0, win);
// 		MPI_Win_unlock(1, win);
// 		assert(*ptr == mesg_size); // check successful transmition
// 	      }
//       }      
//     }
  MPI_Win_free(&win);
  delete []buffer;
  MPI_Finalize ();
  return 0;
}
