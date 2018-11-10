/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Sequential bucketsort implementation
 To compile    : g++ bucketsort.cpp -o bucketsort
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <time.h>
using namespace std;

#define MAXRANGE 1000

#define min(A,B) ((A>B) ? B : A)
//************************************************
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}
//************************************************
int comp(const void *a, const void *b)
{
  int x = *((int *)a);
  int y = *((int *)b);
  return x-y;
}
//************************************************
void BucketSort (int *data, int N, int numBuckets=2)
{
  int *bucket[numBuckets];
  int len[numBuckets];
  int bucketRange = MAXRANGE/numBuckets+1;

  for(int i=0;i<numBuckets;i++)
  {
    bucket[i] = new int[N];
    len[i]=0;
  }
  
  for(int i=0;i<N;i++)
  {
     int buckNum = data[i] /  bucketRange;
     bucket[buckNum][len[buckNum]++] = data[i];
  }

  for(int i=0;i<numBuckets;i++)
    qsort(bucket[i], len[i], sizeof(int), comp);

  int k=0;
  for(int i=0;i<numBuckets;i++)
     for(int j=0;j<len[i];j++)
        data[k++] = bucket[i][j];

 for(int i=0;i<numBuckets;i++)
    delete[] bucket[i];
}
//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc == 1)
    {
      fprintf (stderr, "%s N\n", argv[0]);
      exit (0);
    }
  int N = atoi (argv[1]);
  int *data = (int *) malloc (N * sizeof (int));
  numberGen (N, 1000, data);

  BucketSort(data, N);

//   for(int i=0;i<N;i++)
//     cout << data[i] << " ";
//   cout << endl;
  
  free (data);
  return 0;
}
