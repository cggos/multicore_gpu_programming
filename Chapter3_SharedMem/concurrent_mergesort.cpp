/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake concurrent_mergesort.pro; make
 ============================================================================
 */
// Parallel mergesort, using QtConcurrent functionality
// G. Barlas, Nov. 2013
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <QRunnable>
#include <QThreadPool>
#include <QtConcurrentMap>
#include <QVector>
#include <QTime>

using namespace std;

#define MAXVALUE 10000
#define min(A,B) ((A<B) ? A : B)
#define max(A,B) ((A>B) ? A : B)

//************************************************
template < class T > class ArrayRange
{
private:
  T * store;                    // where data reside
  T *temp;                      // where they can be shifted for sorting
  int start, end;               // end points 1 spot further than the last item.
public:
  ArrayRange (T * x, T * y, int s, int e);
  static void merge (ArrayRange * x, ArrayRange * y);
  static void sort (ArrayRange * x);
  int size ();
  T *getStore ()
  {
    return store;
  }
  void switchStore ();
  void switchPointers ();
};

//------------------------------------
template < class T > int ArrayRange < T >::size ()
{
  return end - start;
}

//------------------------------------
template < class T > void ArrayRange < T >::switchStore ()
{
  for (int i = start; i < end; i++)
    temp[i] = store[i];
  switchPointers ();
}

//------------------------------------
// Only swaps the pointers
template < class T > void ArrayRange < T >::switchPointers ()
{
  T *aux;
  aux = temp;
  temp = store;
  store = aux;
}

//------------------------------------
template < class T > ArrayRange < T >::ArrayRange (T * x, T * y, int s, int e)
{
  store = x;
  temp = y;
  start = s;
  end = e;
}

//------------------------------------
template < class T > void ArrayRange < T >::merge (ArrayRange * x, ArrayRange * y)
{
  // make any initial copy necessary so that the data end-up in the same array
  if (x->store != y->store)
    {
      // determine which is smaller
      int xlen = x->end - x->start;
      int ylen = y->end - y->start;

      if (xlen > ylen)
        y->switchStore ();
      else
        x->switchStore ();
    }

  // now perform merge-list
  int idx1 = x->start, idx2 = y->start;
  int loc = min (idx1, idx2);   // starting point in temp array
  while (idx1 != x->end && idx2 != y->end)
    {
      if (x->store[idx1] <= y->store[idx2])
        {
          x->temp[loc] = x->store[idx1];
          idx1++;
        }
      else
        {
          x->temp[loc] = x->store[idx2];        // same as y->store[idx2]
          idx2++;
        }
      loc++;
    }

  // copy the rest
  for (int i = idx1; i < x->end; i++)
    x->temp[loc++] = x->store[i];

  for (int i = idx2; i < y->end; i++)
    x->temp[loc++] = x->store[i];

  x->start = min (x->start, y->start);
  x->end = max (x->end, y->end);

  // the sorted "stuff" are in temp now  
  x->switchPointers ();
}

//------------------------------------
int comp (const void *a, const void *b)
{
  int x = *((int *) a);
  int y = *((int *) b);
  return x - y;
}

//------------------------------------
template < class T > void ArrayRange < T >::sort (ArrayRange * x)
{
  qsort (x->store + x->start, x->end - x->start, sizeof (T), comp);
}

//************************************************
template < class T > class MergeTask:public QRunnable
{
private:
  ArrayRange < T > *part1;
  ArrayRange < T > *part2;

public:
MergeTask (ArrayRange < T > *p1, ArrayRange < T > *p2):part1 (p1), part2 (p2)
  {
  }
  void run ();
};

//------------------------------------
template < class T > void MergeTask < T >::run ()
{
  ArrayRange < T >::merge (part1, part2);
}

//************************************************
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//--------------------------------------------------------
template < class T > void concurrentMergesort (T * data, int N, int numBlocks = -1)
{
  if (numBlocks < 0)
    numBlocks = 2 * sysconf (_SC_NPROCESSORS_ONLN);

  T *temp = new T[N];

  // 1st step : block setup
  QVector < ArrayRange < T > *>b;
  int pos = 0;
  int len = ceil (N * 1.0 / numBlocks);
  for (int i = 0; i < numBlocks - 1; i++)
    {
      b.append (new ArrayRange < T > (data, temp, pos, pos + len));
      pos += len;
    }
  // setup last block
  b.append (new ArrayRange < T > (data, temp, pos, N));

  // 2nd step : sort the individual blocks concurrently
  QtConcurrent::blockingMap (b, ArrayRange < T >::sort);

  //3rd step: "mergelisting" the pieces
  // merging is done in lg(numBlocks) phases in a bottom-up fashion
  for (int blockDistance = 1; blockDistance < numBlocks; blockDistance *= 2)
    {
      for (int startBlock = 0; startBlock < numBlocks - blockDistance; startBlock += 2 * blockDistance)
        {
          QThreadPool::globalInstance ()->start (new MergeTask < T > (b[startBlock], b[startBlock + blockDistance]));
        }
      // barrier
      QThreadPool::globalInstance ()->waitForDone ();
    }

  // b[0]->store points to the sorted data
  if (b[0]->getStore () != data)        // need to copy data from temp -> data array
    b[0]->switchStore ();
  delete[]temp;
}

//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc < 3)
    {
      cout << "Use : " << argv[0] << " numData numBlocks\n";
      exit (1);
    }

  int N = atoi (argv[1]);
  int *data = new int[N];
  numberGen (N, MAXVALUE, data);

  int numBlocks = atoi (argv[2]);
  QTime t;
  t.start();
  concurrentMergesort (data, N, numBlocks);
  // printout sorting time in msec
  cout << t.elapsed() << endl;
  
  // Sanity check
  for (int i = 0; i < N - 1; i++)
    if (data[i + 1] < data[i])
      cout << "ERROR!\n";

  delete[]data;
  return 0;
}
