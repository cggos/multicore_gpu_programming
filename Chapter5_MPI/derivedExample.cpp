/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC derivedExample.cpp -o derivedExample
 ============================================================================
 */
#include <mpi.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

struct Pixel
{
  int x;
  int y;
  unsigned char RGB[3];
};

void deriveType (MPI_Datatype * t)
{
  struct Pixel sample;

  int blklen[3];
  MPI_Aint displ[3], off, base;
  MPI_Datatype types[3];

  blklen[0] = 1;
  blklen[1] = 1;
  blklen[2] = 3;

  types[0] = MPI_INT;
  types[1] = MPI_INT;
  types[2] = MPI_UNSIGNED_CHAR;

  displ[0] = 0;
  MPI_Get_address (&(sample.x), &base);
  MPI_Get_address (&(sample.y), &off);
  displ[1] = off-base;
  MPI_Get_address (&(sample.RGB[0]), &off);
  displ[2] = off - base;

  MPI_Type_create_struct (3, blklen, displ, types, t);
  MPI_Type_commit (t);
}

int main (int argc, char *argv[])
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;
  MPI_Datatype pixelStr;

  deriveType (&pixelStr);
  if (rank == 0)
    {
      struct Pixel test;
      test.x = 100;
      test.y = 200;
      test.RGB[0] = 10;
      test.RGB[1] = 20;
      test.RGB[2] = 30;
      MPI_Send (&test, 1, pixelStr, 1, 0, MPI_COMM_WORLD);
    }
  else
    {
      struct Pixel test;
      MPI_Recv (&test, 1, pixelStr, 0, 0, MPI_COMM_WORLD, &status);
      cout << test.x << " " << test.y << " " << (int)test.RGB[0] << " " << (int)test.RGB[1] <<" " << (int)test.RGB[2] <<endl;
    }
  MPI_Finalize ();
  return 0;
}
