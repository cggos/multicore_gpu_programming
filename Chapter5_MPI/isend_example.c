/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc isend_example.c -o isend_example
 ============================================================================
 */
#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define RANGEMIN 0
#define RANGEMAX 1000
#define MSGTAG 0

int main (int argc, char **argv)
{
  int rank, num, i;
  int range[2];
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  MPI_Status status;
  
  if (rank == 0)
    {
       MPI_Request rq[num-1];
       int rng[2*num];
       int width = (RANGEMAX - RANGEMIN) / num;
       rng[0] = RANGEMIN;             // left limit
       rng[1] = rng[0] + width - 1; // right limit
       for(i=1;i<num;i++)
          {
            rng[i*2] = rng[i*2-1] + 1;
            rng[i*2+1] = (i==num-1) ? RANGEMAX : rng[i*2] + width - 1;
          }
         
       for(i=1;i<num;i++)
            MPI_Isend(rng+i*2, 2, MPI_INT, i, MSGTAG, MPI_COMM_WORLD, &(rq[i-1]));
          
       for(i=1;i<num;i++)
            MPI_Wait(&(rq[i-1]), &status);
          
     range[0] = rng[0];           // master's limits
     range[1] = rng[1];
    }
  else
    {
      MPI_Request rq;
      MPI_Irecv (range, 2, MPI_INT, 0, MSGTAG, MPI_COMM_WORLD,&rq);
      MPI_Wait(&rq, &status);
    }

  printf ("Node %i's range : ( %i, %i )\n", rank, range[0], range[1]);
    
  MPI_Finalize ();
  return 0;
}
