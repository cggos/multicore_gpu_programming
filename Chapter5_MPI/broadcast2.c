/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc broadcast2.c -o broadcast2
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

    int rank, num;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Status status;

    if (rank == 0) {
        double data;
        scanf("%lf", &data);
        MPI_Bcast(&data, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {

        double data;
        MPI_Bcast(&data, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("Node #%i received %lf\n", rank, data);
    }
    MPI_Finalize();
    return 0;
}
