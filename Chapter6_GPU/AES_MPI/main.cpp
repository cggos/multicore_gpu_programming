/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : February 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <ctime>

#include <mpi.h>

#include "rijndael.h"

using namespace std;

#define TAG_RES  0
#define TAG_WORK 1
#define TAG_DATA 2

static const int keybits = 256;

//******************************************************************************************************
int main (int argc, char *argv[])
{
  timeval timeMain;
  gettimeofday (&timeMain, NULL);
  double tm1 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);
  int rank;
  unsigned char *iobuf;

  int lSize = 0;
  FILE *f;

  int comm_size = 0;
  MPI_Status status;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);
  MPI_Request req;
  MPI_Status stat;


  if (argc < 5)
    {
      if (rank == 0)
        fprintf (stderr, "Usage : %s inputfile outputfile workItemSize threadsPerBlock\n", argv[0]);

      exit (1);
    }

  //encryption key
  unsigned char key[32] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
  u32 rk[RKLENGTH (keybits)];
  rijndaelSetupEncrypt (rk, key, keybits);

  if (rank == 0)
    {
      if ((f = fopen (argv[1], "r")) == NULL)
        {
          fprintf (stderr, "Can't open %s\n", argv[1]);
          exit (EXIT_FAILURE);
        }
       
      int workItemSize = atoi (argv[3]);

      fseek (f, 0, SEEK_END);
      lSize = ftell (f);
      rewind (f);

      iobuf = new unsigned char[lSize];
      assert (iobuf != NULL);
      fread (iobuf, 1, lSize, f);
      fclose (f);

      timeval tim;
      gettimeofday (&tim, NULL);
      double tm2 = tim.tv_sec + (tim.tv_usec / 1000000.0);
      
      // master main loop
      int pos = 0;
      while (pos < lSize)       
        {
          int retPos;
          MPI_Recv (&retPos, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RES, MPI_COMM_WORLD, &stat);
          if (retPos >= 0)      // if not the first dummy worker call
            MPI_Recv (iobuf + retPos, workItemSize, MPI_UNSIGNED_CHAR, stat.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD, &stat);

          // assign next work item
	  int actualSize = (workItemSize < lSize - pos) ? workItemSize : (lSize - pos);
          MPI_Send (&pos, 1, MPI_INT, stat.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
          MPI_Send (iobuf + pos, actualSize, MPI_UNSIGNED_CHAR, stat.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD);
// printf("Assigning %i to %i\n", pos, stat.MPI_SOURCE);
          pos += actualSize;
        }

// printf("Assigned %i from %i\n", pos, lSize);

      // wait for last results
      pos = -1;
      for (int i = 1; i < comm_size; i++)
        {
          int retPos;
          MPI_Recv (&retPos, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RES, MPI_COMM_WORLD, &stat);
          if (retPos >= 0)      // if not the first dummy worker call
            MPI_Recv (iobuf + retPos, workItemSize, MPI_UNSIGNED_CHAR, stat.MPI_SOURCE, TAG_DATA, MPI_COMM_WORLD, &stat);

          // indicate end of operations
          MPI_Send (&pos, 1, MPI_INT, stat.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
        }
      gettimeofday (&tim, NULL);
      double tm3 = tim.tv_sec + (tim.tv_usec / 1000000.0);
      
      FILE *fout;
      if ((fout = fopen (argv[2], "w")) == NULL)
        {
          fprintf (stderr, "Can't open %s\n", argv[2]);
          exit (EXIT_FAILURE);
        }
      fwrite(iobuf, 1, lSize, fout);
      fclose (fout);


      gettimeofday (&timeMain, NULL);
      double tm4 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

      // print-out some timing information
      printf ("%.9lf \t %.9lf \n", tm4-tm1, tm3-tm2);
    }
  else                          // GPU worker
    {      
      int workItemSize = atoi (argv[3]);
      int thrPerBlock = atoi (argv[4]);
      int pos = -1;
      int totalWork=0;
      iobuf = new unsigned char[workItemSize];
      
      MPI_Send (&pos, 1, MPI_INT, 0, TAG_RES, MPI_COMM_WORLD);
      MPI_Recv (&pos, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, &stat);
      while (pos >= 0)
        {
          MPI_Recv (iobuf, workItemSize, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD, &stat);
	  int actualSize;
	  MPI_Get_count(&stat, MPI_UNSIGNED_CHAR, &actualSize);
          totalWork+=actualSize;
          rijndaelEncryptFE (rk, keybits, iobuf, iobuf, actualSize, thrPerBlock);
          MPI_Send (&pos, 1, MPI_INT, 0, TAG_RES, MPI_COMM_WORLD);
          MPI_Send (iobuf, actualSize, MPI_UNSIGNED_CHAR, 0, TAG_DATA, MPI_COMM_WORLD);

          // get next work item start            
          MPI_Recv (&pos, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, &stat);
        }
      rijndaelShutdown();
      cout << "Worker " << rank << " processed " << totalWork << endl;
    }

  MPI_Finalize ();

  delete[]iobuf;
  return 0;
}
