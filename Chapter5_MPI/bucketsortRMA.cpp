/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Parallel bucket sort, with RMA 
 To build use  : mpiCC bucketsortRMA.cpp -o bucketsortRMA
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
void dump(int *d, int M)
{
  for (int i = 0; i < M; i++)
    cout << d[i] << " " ;
  cout << endl;
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
  int *data = new int[M];
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

  // three windows created, one for the bucket counts, one for the buckets themselves and one for the data
  MPI_Group all, allOtherGroup;
  MPI_Comm_group (MPI_COMM_WORLD, &all);
  MPI_Group_excl (all, 1, &rank, &allOtherGroup);
  MPI_Win cntWin, bucketWin, dataWin;
  MPI_Win_create (buckets, N * maxItemsPerBucket * sizeof (int), sizeof (int), MPI_INFO_NULL, MPI_COMM_WORLD, &bucketWin);
  MPI_Win_create (inBucket, N * sizeof (int), sizeof (int), MPI_INFO_NULL, MPI_COMM_WORLD, &cntWin);
  MPI_Win_create (data, M * sizeof (int), sizeof (int), MPI_INFO_NULL, MPI_COMM_WORLD, &dataWin);

  // step 1
  // replacing MPI_Scatter (data, maxItemsPerBucket, MPI_INT, data, maxItemsPerBucket, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Win_fence (0, dataWin);
  if (rank == 0)
    {
      for (int i = 1; i < N; i++)
        {
          deliveredItems = (i == N - 1) ? (M - (N - 1) * maxItemsPerBucket) : maxItemsPerBucket;
          MPI_Put (&(data[bucketOffset[i]]), deliveredItems, MPI_INT, i, 0, N * maxItemsPerBucket, MPI_INT, dataWin);
        }
    }
  MPI_Win_fence (0, dataWin);
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
  // replacing MPI_Alltoall (inBucket, 1, MPI_INT, toRecv, 1, MPI_INT, MPI_COMM_WORLD);
  toRecv[rank] = inBucket[rank];
  MPI_Win_post (allOtherGroup, 0, cntWin);
  MPI_Win_start (allOtherGroup, 0, cntWin);
  for (int i = 0; i < N; i++)
    if (i != rank)
      MPI_Get (&(toRecv[i]), 1, MPI_INT, i, rank , 1, MPI_INT, cntWin);
  MPI_Win_complete (cntWin);
  MPI_Win_wait (cntWin);

  recvOff[0] = 0;
  for (int i = 1; i < N; i++)
    recvOff[i] = recvOff[i - 1] + toRecv[i - 1];

  // replacing MPI_Alltoallv (buckets, inBucket, bucketOffset, MPI_INT, data, toRecv, recvOff, MPI_INT, MPI_COMM_WORLD);
  MPI_Win_post (all, 0, bucketWin);
  MPI_Win_start (all, 0, bucketWin);
  for (int i = 0; i < N; i++)
    MPI_Get (&(data[recvOff[i]]), toRecv[i], MPI_INT, i, bucketOffset[rank], maxItemsPerBucket, MPI_INT, bucketWin);
  MPI_Win_complete (bucketWin);
  MPI_Win_wait (bucketWin);

  MPI_Win_lock (MPI_LOCK_EXCLUSIVE, rank, 0, dataWin);  // limit access to data array until it is sorted

  
  // step 4
  // apply quicksort to the local bucket
  int localBucketSize = recvOff[N - 1] + toRecv[N - 1];
  qsort (data, localBucketSize, sizeof (int), comp);

  MPI_Win_unlock (rank, dataWin);       // data array is available again

  // step 5
  MPI_Gather (&localBucketSize, 1, MPI_INT, toRecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    {
      recvOff[0] = 0;
      for (int i = 1; i < N; i++)
        recvOff[i] = recvOff[i - 1] + toRecv[i - 1];
    }

  // replacing MPI_Gatherv (data, localBucketSize, MPI_INT, data, toRecv, recvOff, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    {
      for (int i = 1; i < N; i++)
        {
          MPI_Win_lock (MPI_LOCK_EXCLUSIVE, i, 0, dataWin);     // gain access to remote data array
          MPI_Get (&(data[recvOff[i]]), toRecv[i], MPI_INT, i, 0, toRecv[i], MPI_INT, dataWin);
          MPI_Win_unlock (i, dataWin);  // release lock to remote data array
        }
    }

  // print results
  if (rank == 0)
    {
      for (int i = 0; i < M; i++)
        cout << data[i] << " ";
      cout << endl;
    }

  MPI_Group_free (&all);
  MPI_Group_free (&allOtherGroup);
  MPI_Win_free (&cntWin);
  MPI_Win_free (&bucketWin);
  MPI_Win_free (&dataWin);
  MPI_Finalize ();
  delete[]buckets;
  delete[]inBucket;
  delete[]bucketOffset;
  delete[]toRecv;
  delete[]recvOff;
  delete[]data;
  return 0;
}
