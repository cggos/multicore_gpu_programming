/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC packUnpack.cpp -o packUnpack
 ============================================================================
 */
#include <mpi.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

struct Pixel
{
  int x, y;
  unsigned char RGB[3];
};

int main (int argc, char *argv[])
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;

  unsigned char *buffer = new unsigned char[100];
  if (rank == 0)
    {
      struct Pixel test;
      test.x = 100;
      test.y = 200;
      test.RGB[0] = 10;
      test.RGB[1] = 20;
      test.RGB[2] = 30;
      
      // pack everything up
      int position=0;
      MPI_Pack(&(test.x), 1, MPI_INT, buffer, 100, &position, MPI_COMM_WORLD);
      MPI_Pack(&(test.y), 1, MPI_INT, buffer, 100, &position, MPI_COMM_WORLD);
      MPI_Pack(test.RGB, 3, MPI_UNSIGNED_CHAR, buffer, 100, &position, MPI_COMM_WORLD);
      MPI_Send (buffer, position, MPI_UNSIGNED_CHAR, 1, 0, MPI_COMM_WORLD);
    }
  else
    {
      struct Pixel test;
      int position=0;
      MPI_Recv (buffer, 100, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, &status);

      // now start unpacking
      MPI_Unpack(buffer, 100, &position, &(test.x), 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(buffer, 100, &position, &(test.y), 1, MPI_INT, MPI_COMM_WORLD);
      MPI_Unpack(buffer, 100, &position, test.RGB, 3, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
      
      cout << test.x << " " << test.y << " " << (int)test.RGB[0] << " " << (int)test.RGB[1] <<" " << (int)test.RGB[2] <<endl;
    }
  delete [] buffer;
  MPI_Finalize ();
  return 0;
}
