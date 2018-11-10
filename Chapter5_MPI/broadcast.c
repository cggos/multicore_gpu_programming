/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpicc broadcast.c -o broadcast
 ============================================================================
 */
#include<mpi.h>
#include<string.h>
#include<stdio.h>
#define MESSTAG 0
#define MAXLEN 100

//*****************************************
// Returns the position of the most significant set bit of its argument
int MSB(int i) {
    int pos = 0;
    while (i != 0) {
        i >>= 1;
        pos++;
    }
    return pos-1;
}
//*****************************************

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, num, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Status status;

    if (rank == 0) {
        int destID = 1;
        double data;
        scanf("%lf", &data);
        while (destID < num) { // a subset of nodes gets a message
            MPI_Send(&data, 1, MPI_DOUBLE, destID, MESSTAG, MPI_COMM_WORLD);
            destID <<= 1;
        }
    } else {
        int msbPos = MSB(rank);
        int srcID = rank ^ (1 << msbPos); // the message is not coming from 0 for all
        printf("#%i has source %i\n", rank, srcID);

        double data;
        MPI_Recv(&data, 1, MPI_DOUBLE, srcID, MESSTAG, MPI_COMM_WORLD, &status);
        printf("Node #%i received %lf\n", rank, data);

        // calculate the ID of the node that will receive a copy of the message
        int destID = rank | (1 << (msbPos + 1));
        while (destID < num) {
            MPI_Send(&data, 1, MPI_DOUBLE, destID, MESSTAG, MPI_COMM_WORLD);
            msbPos++;
            destID = rank | (1 << (msbPos + 1));
        }
    }
    MPI_Finalize();
    return 0;
}
