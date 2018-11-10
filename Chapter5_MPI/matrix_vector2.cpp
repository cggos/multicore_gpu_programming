/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Matrix-Vector Multiplication, using MPI_Scatterv
 To build use  : mpiCC matrix_vector2.cpp -o matrix_vector2
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
  rowsPerProcess = M/N;

  if (rank == 0)
    {
      double *A = new double[M * M];
      double *B = new double[M];
      double *C = new double[M];        // result vector

      for (int i = 0; i < M * M; i++)    A[i] = i;
      for (int i = 0; i < M; i++)        B[i] = 1;

      int displs[N];
      int sendcnts[N];
      for(int i=0;i<N;i++)
      {
	  sendcnts[i] = rowsPerProcess*M;
	  displs[i] = i*rowsPerProcess*M;
	  if(i==N-1)
  	    sendcnts[i] = (M - (N-1)*rowsPerProcess)*M;	    
	  cout << "To " << i << " sending " << sendcnts[i] << endl;
      }
      
      MPI_Scatterv (A, sendcnts,displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
      if (rank == N - 1)
        rowsPerProcess = M - (N - 1) * rowsPerProcess;  
      double *locA = new double[M * rowsPerProcess];
      double *B = new double[M];
      double *partC = new double[rowsPerProcess];       // partial result vector

      MPI_Scatterv (NULL, NULL, NULL, MPI_DOUBLE, locA, M * rowsPerProcess, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast (B, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MV (locA, B, partC, M, rowsPerProcess);
      MPI_Send (partC, rowsPerProcess, MPI_DOUBLE, 0, RESTAG, MPI_COMM_WORLD);
    }

  MPI_Finalize ();
  return 0;
}
