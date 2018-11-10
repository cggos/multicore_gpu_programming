/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Assumes that node 0 is a CPU node
                 Runs CPU code on the workers too
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
#include <limits.h>
#include <ctime>

#include <mpi.h>

#include "rijndael.h"
#include "partition.cpp"

using namespace std;

#define TAG_MODEL 0
#define TAG_WORK 1

const double modelParams[]={6.50412749748771E-009, 413260.323106647}; // p_i and e_i
const double l= 1.2441013041013E-010;
const double b= 7408.7996426914;
const double d= 7408.7996426914;

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

  if (argc < 4)
    {
      if (rank == 0)
        fprintf (stderr, "Usage : %s inputfile outputfile threadsPerBlock\n", argv[0]);

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
       
      fseek (f, 0, SEEK_END);
      lSize = ftell (f);
      rewind (f);

      timeval tim;
      gettimeofday (&tim, NULL);
      double tm2 = tim.tv_sec + (tim.tv_usec / 1000000.0);
      
      
      // allocate the arrays needed for calculating the partitioning
      double *p=new double[comm_size];
      double *e=new double[comm_size];
      double *part=new double[comm_size];
      long *assignment = new long[comm_size];
      long *offset = new long[comm_size];
      long *indLenPairs = new long[comm_size*2];
      MPI_Request *rq = new  MPI_Request[comm_size]; 
      
      // get the characteristics of each node
      p[0] = modelParams[0];
      e[0] = modelParams[1];
      double temp[2];
      for(int i=1;i<comm_size;i++)
      {
         MPI_Recv (temp, 2, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_MODEL, MPI_COMM_WORLD, &stat);
         int idx = stat.MPI_SOURCE;
	 p[idx] = temp[0];
	 e[idx] = temp[1];
      }
      // calculate the assignment. Communication speed is divided between the N-1 workers
      double predTime = nPortPartition(p, e, lSize, l*(comm_size-1), b, d, part, comm_size);
      
      quantize(part, lSize, 16, comm_size, assignment);

      // calculate the start_offset, length of the assignment parts
      long pos = 0;
      for(int i=0;i<comm_size;i++)
      {
	  indLenPairs[2*i] = pos;
	  indLenPairs[2*i+1] = assignment[i];
	  pos +=assignment[i];
      }      
     
      // communicate the assigned plaintext start_off, length pairs
      for(int i=1;i<comm_size;i++)
          MPI_Isend (indLenPairs +2*i, 2, MPI_LONG, i, TAG_WORK, MPI_COMM_WORLD, &req);
      
      
      // process part0 of the input on the CPU     
      int nrounds = NROUNDS (keybits);
      int dataPos;

      iobuf = new unsigned char[assignment[0]];
      assert (iobuf != NULL);
      fread (iobuf, 1, assignment[0], f);
      fclose (f);

      
      for (dataPos = 0; dataPos < assignment[0]; dataPos += 16)
            {
              // encrypt 16-byte block
              rijndaelCPUEncrypt (rk, nrounds, iobuf + dataPos, iobuf + dataPos);
            }      
  
      
      gettimeofday (&tim, NULL);
      double tm3 = tim.tv_sec + (tim.tv_usec / 1000000.0);
      
      FILE *fout;
      if ((fout = fopen (argv[2], "w")) == NULL)
        {
          fprintf (stderr, "Can't open %s\n", argv[2]);
          exit (EXIT_FAILURE);
        }
      fwrite(iobuf, 1, assignment[0], fout);
      fclose (fout);


      // wait for all to finish
      MPI_Barrier(MPI_COMM_WORLD);   

      gettimeofday (&timeMain, NULL);
      double tm4 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

      // print-out some timing information
      printf ("Predicted %.9lf \t Measured : %.9lf \t %.9lf \n", predTime, tm4-tm1, tm3-tm2);     

      delete[] p;
      delete[] e;
      delete[] part;
      delete[] assignment;
      delete[] offset;
      delete[] rq;
      delete[] indLenPairs;
    }
  else                          // CPU worker
    {      
      
      long indLenPairs[2];
      
      // send model parameters
      MPI_Send ((void*)modelParams, 2, MPI_DOUBLE, 0, TAG_MODEL, MPI_COMM_WORLD);
      
      gettimeofday (&timeMain, NULL);
      double wt1 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);
      
      // get size of assignment and allocate appropriate buffer
      MPI_Recv (indLenPairs, 2, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, &stat);

      long jobSize = indLenPairs[1];
      iobuf = new unsigned char[jobSize];
      FILE *f;
      if ((f = fopen (argv[1], "r")) == NULL)
        {
          fprintf (stderr, "Can't open %s\n", argv[1]);
          exit (EXIT_FAILURE);
        }
      fseek(f,indLenPairs[0], SEEK_SET);  
      fread(iobuf, 1, indLenPairs[1], f);
      fclose (f);
      

      gettimeofday (&timeMain, NULL);
      double wt2 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

      int nrounds = NROUNDS (keybits);
      int dataPos;
      for (dataPos = 0; dataPos < jobSize; dataPos += 16)
            {
              // encrypt 16-byte block
              rijndaelCPUEncrypt (rk, nrounds, iobuf + dataPos, iobuf + dataPos);
            }
      gettimeofday (&timeMain, NULL);
      double wt3 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

      FILE *fout;
      if ((fout = fopen (argv[2], "w")) == NULL)
        {
          fprintf (stderr, "Can't open %s\n", argv[2]);
          exit (EXIT_FAILURE);
        }
      fseek(f,indLenPairs[0], SEEK_SET);  
      fwrite(iobuf, 1, jobSize, fout);
      fclose (fout);
      
      gettimeofday (&timeMain, NULL);
      double wt4 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

      MPI_Barrier(MPI_COMM_WORLD);
//       cout << "Worker " << rank << " Distr:" << wt2-wt1 << " Comp:" << wt3-wt2 << " Coll:" << wt4-wt3 << endl;
      rijndaelShutdown();
      
    }

  MPI_Finalize ();

  delete[]iobuf;
  return 0;
  
}
