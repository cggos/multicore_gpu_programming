/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc worker.c -o worker
 ============================================================================
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAXLEN 100
char *greetings[] = { "Hello", "Hi", "Awaiting your command" };
char buff[MAXLEN];

int main (int argc, char **argv)
{
  srand (time (0));
  MPI_Init (&argc, &argv);
  int grID = rand () % 3;
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  sprintf (buff, "Node %i says %s", rank, greetings[grID]);
  MPI_Send (buff, strlen (buff), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
  MPI_Finalize ();
}
