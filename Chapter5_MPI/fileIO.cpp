/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Block distribution of data using MPI I/O
 To build use  : mpiCC fileIO.cpp -o fileIO
 ============================================================================
 */
#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#include<unistd.h>
#include<iostream>

using namespace std;

//*****************************************
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;

  if (argc == 1)
    {
      if (rank == 0)
        cerr << "Usage " << argv[0] << " filetoload\n";
      exit (1);
    }

  MPI_File f;
  MPI_File_open (MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);


  int *data;
  int blockSize;

  MPI_Offset filesize;
  MPI_File_get_size (f, &filesize); // get file size in bytes
  filesize /= sizeof(int);          // convert to number of items
  blockSize = filesize / N;         // calculate size of block to read per process
  int pos = rank * blockSize;       // initial file position per process
  if(rank == N-1)
      blockSize = filesize - pos;   // get all remaining in last process
      
  data = new int[blockSize];
  MPI_File_seek(f, pos*sizeof(int), MPI_SEEK_SET);
  MPI_File_read (f, data, blockSize, MPI_INT, &status);
  MPI_File_close (&f);

  sleep (rank);
  cout << rank << " read " << blockSize << " numbers." << endl;
  for (int i = 0; i < 30; i++)
    cout << data[i] << " ";
  cout << ".... Last one is : " << data[blockSize - 1];
  cout << endl;

  delete [] data;
  MPI_Finalize ();
  return 0;
}
