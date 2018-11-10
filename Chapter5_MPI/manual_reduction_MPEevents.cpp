/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Reduction implementation with MPE events
 To build use  : mpecc manual_reduction_MPEevents.cpp -mpilog -lmpi_cxx -lstdc++ -o manual_reduction_MPEevents
 ============================================================================
 */
#include<mpi.h>
#include<mpe.h>
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

  // calculate how many events will be monitored
  int numEvents = 0;
  int bitMask = 1;
  while (bitMask < N)
    {
      numEvents++;
      bitMask <<= 1;
    }
  int startID[numEvents];
  int finalID[numEvents];
  char *eventNames[numEvents];
  for (int i = 0; i < numEvents; i++)
    {
      MPE_Log_get_state_eventIDs (&(startID[i]), &(finalID[i]));
      eventNames[i] = new char[10];
      sprintf (eventNames[i], "Phase%i\0", i);
      MPE_Describe_state (startID[i], finalID[i], eventNames[i], "red");
    }

  int partialSum = rank;
  bitMask = 1;
  bool doneFlag = false;
  int phase = 0;
  while (bitMask < N && !doneFlag)
    {
      int otherPartyID;
      if ((rank & bitMask) == 0)
        {
          otherPartyID = rank | bitMask;
          if (otherPartyID >= N)
            {
              bitMask <<= 1;
              continue;
            }
          MPE_Log_event (startID[phase], 0, NULL);

          int temp;
          MPI_Recv (&temp, 1, MPI_INT, otherPartyID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          partialSum += temp;
          MPE_Log_event (finalID[phase], 0, NULL);
        }
      else
        {
          MPE_Log_event (startID[phase], 0, NULL);
          otherPartyID = rank ^ bitMask;
          doneFlag = true;
          MPI_Send (&partialSum, 1, MPI_INT, otherPartyID, 0, MPI_COMM_WORLD);
          MPE_Log_event (finalID[phase], 0, NULL);
        }
      bitMask <<= 1;
      phase++;
    }

  if (rank == 0)
    cout << partialSum << endl;

  MPI_Finalize ();
  return 0;
}
