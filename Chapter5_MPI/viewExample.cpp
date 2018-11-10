/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC viewExample.cpp -o viewExample
 ============================================================================
 */
// Cyclic block distribution of data using MPI I/O
#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>
#include <vector>
#include <unistd.h>

using namespace std;

const int BLOCKSIZE = 10;

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

  MPI_Datatype filetype;
  int sizes = N * BLOCKSIZE, subsizes = BLOCKSIZE, starts = 0;
  MPI_Type_create_subarray (1, &sizes, &subsizes, &starts, MPI_ORDER_C, MPI_INT, &filetype);
  MPI_Type_commit (&filetype);
  MPI_File_set_view (f, rank * BLOCKSIZE * sizeof (int), MPI_INT, filetype, "native", MPI_INFO_NULL);

  vector < int >data;
  int temp[BLOCKSIZE];

  MPI_Offset filesize;
  MPI_File_get_size (f, &filesize); // get size in bytes
  filesize /= sizeof (int);         // convert size in number of items
  int pos = rank * BLOCKSIZE;       // initial file position per process
  while (pos < filesize)
    {
      MPI_File_read (f, temp, 1, filetype, &status);
      int cnt;
      MPI_Get_count (&status, filetype, &cnt);

      pos += BLOCKSIZE * N;
      for (int i = 0; i < cnt * BLOCKSIZE; i++)
        data.push_back (temp[i]);
    }

  MPI_File_close (&f);

  sleep (rank);
  cout << rank << " read " << data.size () << " numbers." << endl;
  for (int i = 0; i < 30; i++)
    cout << data[i] << " ";
  cout << ".... Last one is : " << data[data.size () - 1];
  cout << endl;

  MPI_Finalize ();
  return 0;
}
