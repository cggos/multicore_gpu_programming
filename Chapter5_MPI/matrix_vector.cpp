/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC matrix_vector.cpp -o matrix_vector
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>
#include<iostream>
#define RESTAG 0

using namespace std;
const int M = 100;

//*****************************************
void MV (double *A, double *B, double *C, int columns, int rows)
{
  for (int i = 0; i < rows; i++)
    {
      double temp = 0;
      for (int j = 0; j < columns; j++)
        temp += A[i * columns + j] * B[j];

      C[i] = temp;
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


  int rowsPerProcess;           // size of block per MPI process
  int rowsAlloc = M;
  if (M % N != 0)
    rowsAlloc = (M / N + 1) * N;
  rowsPerProcess = rowsAlloc / N;

  if (rank == 0)
    {
      double *A = new double[M * rowsAlloc];
      double *B = new double[M];
      double *C = new double[M];        // result vector

      for (int i = 0; i < M * M; i++)
        A[i] = i;
      for (int i = 0; i < M; i++)
        B[i] = 1;

      MPI_Scatter (A, M * rowsPerProcess, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast (B, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MV (A, B, C, M, rowsPerProcess);
      for (int i = 1; i < N - 1; i++)
        MPI_Recv (C + rowsPerProcess * i, rowsPerProcess, MPI_DOUBLE, i, RESTAG, MPI_COMM_WORLD, &status);

      // last process treated differently
      MPI_Recv (C + rowsPerProcess * (N - 1), M - (N - 1) * rowsPerProcess, MPI_DOUBLE, N - 1, RESTAG, MPI_COMM_WORLD, &status);

      for (int i = 0; i < M; i++)
        cout << C[i] << " ";
      cout << endl;
    }
  else
    {
      double *locA = new double[M * rowsPerProcess];
      double *B = new double[M];
      double *partC = new double[rowsPerProcess];       // partial result vector

      MPI_Scatter (NULL, M * rowsPerProcess, MPI_DOUBLE, locA, M * rowsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast (B, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (rank == N - 1)
        rowsPerProcess = M - (N - 1) * rowsPerProcess;  // strip padded rows for the last process
      MV (locA, B, partC, M, rowsPerProcess);
      MPI_Send (partC, rowsPerProcess, MPI_DOUBLE, 0, RESTAG, MPI_COMM_WORLD);
    }

  MPI_Finalize ();
  return 0;
}
