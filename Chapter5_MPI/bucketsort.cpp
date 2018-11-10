/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Parallel bucket sort
 To build use  : mpiCC bucketsort.cpp -o bucketsort
 ============================================================================
 */

#include<mpi.h>
#include<stdlib.h>
#include<math.h>
#include<iostream>

using namespace std;

const int MIN = 0;
const int MAX = 10000;
//*****************************************
int comp (const void *a, const void *b)
{
  return *(reinterpret_cast < const int *>(a)) -*(reinterpret_cast < const int *>(b));
}

//*****************************************
void initData (int min, int max, int *d, int M)
{
  srand (time (0));
  for (int i = 0; i < M; i++)
    d[i] = (rand () % (max - min)) + min;
}

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
        cerr << "Usage " << argv[0] << " number_of_items\n";
      exit (1);
    }

  int M = atoi (argv[1]);
  int maxItemsPerBucket = ceil (1.0 * M / N);
  int deliveredItems;
  int bucketRange = ceil (1.0 * (MAX - MIN) / N);
  int *data = new int[N * maxItemsPerBucket];   // to allow easy scattering
  int *buckets = new int[N * maxItemsPerBucket];
  int *bucketOffset = new int[N];       // where do buckets begin?
  int *inBucket = new int[N];   // how many items in each one?

  int *toRecv = new int[N];     // how many items to receive from each process
  int *recvOff = new int[N];    // offsets for sent data 

  if (rank == 0)
    initData (MIN, MAX, data, M);

  // initialize bucket counters and offsets
  for (int i = 0; i < N; i++)
    {
      inBucket[i] = 0;
      bucketOffset[i] = i * maxItemsPerBucket;
    }


  // step 1
  MPI_Scatter (data, maxItemsPerBucket, MPI_INT, data, maxItemsPerBucket, MPI_INT, 0, MPI_COMM_WORLD);
  deliveredItems = (rank == N - 1) ? (M - (N - 1) * maxItemsPerBucket) : maxItemsPerBucket;

  // step 2
  // split into buckets
  for (int i = 0; i < deliveredItems; i++)
    {
      int idx = (data[i] - MIN) / bucketRange;
      int off = bucketOffset[idx] + inBucket[idx];
      buckets[off] = data[i];
      inBucket[idx]++;
    }

  // step 3
  // start by gathering the counts of data the other processes will send
  MPI_Alltoall (inBucket, 1, MPI_INT, toRecv, 1, MPI_INT, MPI_COMM_WORLD);
  recvOff[0] = 0;
  for (int i = 1; i < N; i++)
    recvOff[i] = recvOff[i - 1] + toRecv[i - 1];

  MPI_Alltoallv (buckets, inBucket, bucketOffset, MPI_INT, data, toRecv, recvOff, MPI_INT, MPI_COMM_WORLD);

  // step 4
  // apply quicksort to the local bucket
  int localBucketSize = recvOff[N - 1] + toRecv[N - 1];
  qsort (data, localBucketSize, sizeof (int), comp);

  // step 5
  MPI_Gather (&localBucketSize, 1, MPI_INT, toRecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    {
      recvOff[0] = 0;
      for (int i = 1; i < N; i++)
        recvOff[i] = recvOff[i - 1] + toRecv[i - 1];
    }

  MPI_Gatherv (data, localBucketSize, MPI_INT, data, toRecv, recvOff, MPI_INT, 0, MPI_COMM_WORLD);

  // print results
  if (rank == 0)
    {
      for (int i = 0; i < M; i++)
        cout << data[i] << " ";
      cout << endl;
    }


  MPI_Finalize ();
  delete[]buckets;
  delete[]inBucket;
  delete[]bucketOffset;
  delete[]toRecv;
  delete[]recvOff;
  delete[]data;
  return 0;
}
