/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.01
 Last modified : January 2015
 License       : Released under the GNU GPL 3.0
 Description   : Matrix-Matrix Multiplication, using MPI_Scatterv and MPI_Type_vector
                 The number of processes must form a grid X*Y that divides the matrices evenly
 To build use  : mpiCC matrixMult.cpp -o matrixMult
 ============================================================================
 */

#include<mpi.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<iostream>
#include<assert.h>

using namespace std;
const int K = 10;
const int M = 10;
const int L = 10;
//*****************************************
void printMatrix (double *m, int r, int c)
{
  for (int i = 0; i < r; i++)
    {
      for (int j = 0; j < c; j++)
        cout << m[i * c + j] << " ";
      cout << endl;
    }
}

//*****************************************
// Sequential code for testing purposes
void MM (double *A, double *B, double *C, int rowsA, int colsA, int colsB)
{
  for (int i = 0; i < rowsA; i++)
    for (int j = 0; j < colsB; j++)
      {
        double temp = 0;
        for (int n = 0; n < colsA; n++)
          temp += A[i * colsA + n] * B[n * colsB + j];

        C[i * colsB + j] = temp;
      }
}

//*****************************************
// Calculates a block of matrix C
// A separate function is needed because the layout of B in process 0 is different from the 
// layout of B in the worker processes. This is reflected in the trueBcols parameter
void MMpartial (double *A, double *B, double *C, int rowsA, int colsA, int colsB, int trueBcols)
{
  for (int i = 0; i < rowsA; i++)
    for (int j = 0; j < colsB; j++)
      {
        double temp = 0;
        for (int n = 0; n < colsA; n++)
          temp += A[i * colsA + n] * B[n * trueBcols + j];

        C[i * trueBcols + j] = temp;
      }
}

//*****************************************
int main (int argc, char **argv)
{
  MPI_Init (&argc, &argv);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Status status;
  MPI_Datatype columnGroup;     // Datatype for communicating B's columns
  MPI_Datatype matrBlock;       // Datatype for communicating C's blocks

  int procX, procY;
  procX = atoi (argv[1]);       // expects a X*Y grid of processes for
  procY = atoi (argv[2]);       // calculating the product
  if (procX * procY != N)       // It will abort if there are not enough processes to form the grid
    MPI_Abort (MPI_COMM_WORLD, 0);

  int rowsPerProcess;           // size of block related to A
  int columnsPerProcess;        // size of block related to B
  rowsPerProcess = K / procY;   // each process will calculate 
  columnsPerProcess = L / procX; // rowsPerProcess*columnsPerProcess elements of C

  // K and L must be multiples of procY and procX respectively
  assert(rowsPerProcess * procY == K && columnsPerProcess * procX == L && N == procX * procY);
  
  if (rank == 0)
    {
      MPI_Type_vector (M, columnsPerProcess, L, MPI_DOUBLE, &columnGroup);
      MPI_Type_commit (&columnGroup);
      MPI_Type_vector (rowsPerProcess, columnsPerProcess, L, MPI_DOUBLE, &matrBlock);
      MPI_Type_commit (&matrBlock);
      double *A = new double[K * M];
      double *B = new double[M * L];
      double *C = new double[K * L];

      // A and B are initialized to values that can be used to check
      // the correctness of the result.
      for (int i = 0; i < K * M; i++)  A[i] = i;
      for (int i = 0; i < M * L; i++)  B[i] = 0;
      for (int i = 0; i < M; i++)      B[i * L + i] = 1;  // B is the identity matrix

      // distribute A first
      int displs[N];
      int sendcnts[N];
      int cntr = 0;
      for (int i = 0; i < procY; i++)
        for (int j = 0; j < procX; j++)
          {
            sendcnts[cntr] = rowsPerProcess * M;
            displs[cntr] = i * rowsPerProcess * M;
            cntr++;
          }

      MPI_Scatterv (A, sendcnts, displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      // now distribute B
      cntr = 1;
      for (int i = 0; i < procY; i++)
        for (int j = 0; j < procX; j++)
          if (i + j != 0)
            {
              MPI_Send (B + j * columnsPerProcess, 1, columnGroup, cntr, 0, MPI_COMM_WORLD);
              cntr++;
            }

      // partial result calculation
      MMpartial (A, B, C, rowsPerProcess, M, columnsPerProcess, L);

      // now collect all the subblocks of C
      cntr = 1;
      for (int i = 0; i < procY; i++)
        for (int j = 0; j < procX; j++)
          if (i + j != 0)
            {
              MPI_Recv (C + i * L * rowsPerProcess + j * columnsPerProcess, 1, matrBlock, cntr, 0, MPI_COMM_WORLD, &status);
              cntr++;
            }

      printMatrix (C, K, L);

    }
  else
    {
      MPI_Type_vector (M, columnsPerProcess, columnsPerProcess, MPI_DOUBLE, &columnGroup);
      MPI_Type_commit (&columnGroup);
      MPI_Type_vector (rowsPerProcess, columnsPerProcess, columnsPerProcess, MPI_DOUBLE, &matrBlock);
      MPI_Type_commit (&matrBlock);

      double *locA = new double[rowsPerProcess * M];
      double *locB = new double[M * columnsPerProcess];
      double *partC = new double[rowsPerProcess * columnsPerProcess];   // partial result matrix

      MPI_Scatterv (NULL, NULL, NULL, MPI_DOUBLE, locA, rowsPerProcess * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      MPI_Recv (locB, 1, columnGroup, 0, 0, MPI_COMM_WORLD, &status);

      MMpartial (locA, locB, partC, rowsPerProcess, M, columnsPerProcess, columnsPerProcess);

      MPI_Send (partC, 1, matrBlock, 0, 0, MPI_COMM_WORLD);
    }

  MPI_Finalize ();
  return 0;
}
