/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc ping_pong.cu -lmpi -I/usr/include/mpi -o ping_pong 
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <cuda.h>

using namespace std;

#define MESG_TAG 0
#define END_TAG 1
#define MAX_MESG 1000000

const int REP = 10;

__global__ void setKernel (char *data)
{

}

int main (int argc, char *argv[])
{
  int size, rank;
  int namelen;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int mesg_size;
  int tag;
  double start_time, end_time;
  MPI_Status status;

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

  int numDevices;
  cudaGetDeviceCount (&numDevices);
  if (numDevices == 0)
    {
      printf ("Node %i does not have a CUDA capable GPU\n", rank);
      exit (1);
    }

  // allocate host and device memory  
  char *d_data, *h_data;
  cudaMalloc ((void **) &d_data, MAX_MESG);
  h_data = (char *) malloc (MAX_MESG);

  if (rank == 0)
    {
      for (int mesg_size = 0; mesg_size <= MAX_MESG; mesg_size += 1000)
        {
          start_time = MPI_Wtime ();
          for (int i = 0; i < REP; i++)
            {
              // get data from device
              cudaMemcpy (h_data, d_data, mesg_size, cudaMemcpyDeviceToHost);
              // send it to other host
              MPI_Send (h_data, mesg_size, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
              // wait to collect response
              MPI_Recv (h_data, MAX_MESG, MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	      // forward data to device
              cudaMemcpy (d_data, h_data, mesg_size, cudaMemcpyHostToDevice);
            }

          end_time = MPI_Wtime ();
          printf ("%i %lf\n", mesg_size, (end_time - start_time) / 2 / REP);
        }

      tag = END_TAG;
      MPI_Send (h_data, 0, MPI_CHAR, 1, tag, MPI_COMM_WORLD);
    }
  else
    {
      while (1)
        {
          // get message
          MPI_Recv (h_data, MAX_MESG, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          // filter-out end tag
          if (status.MPI_TAG == END_TAG)
            break;

          // get size of message and sent it to the device
          MPI_Get_count (&status, MPI_CHAR, &mesg_size);
          cudaMemcpy (d_data, h_data, mesg_size, cudaMemcpyHostToDevice);

          // get response from the device
          cudaMemcpy (h_data, d_data, mesg_size, cudaMemcpyDeviceToHost);
          // and send it back
          MPI_Send (h_data, mesg_size, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
        }
    }

  free (d_data);
  cudaFree (d_data);
  cudaDeviceReset ();
  MPI_Finalize ();
  return 0;
}
