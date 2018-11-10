/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc scatter.c -o scatter
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>
#define MESSTAG 0
#define MAXLEN 100

//*****************************************
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Status status;

    double recvbuf[10];

    if (rank == 0) {
        double data[]={0,1,2,3,4,5,6,7,8,9};
        MPI_Scatter(data, 1, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {

        MPI_Scatter(NULL, 1, MPI_DOUBLE, recvbuf, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
        
        printf("Node #%i received ", rank);
	for(i=0;i<10;i++)
	  printf("%lf ",recvbuf[i]);
	printf("\n");
    MPI_Finalize();
    return 0;
}
