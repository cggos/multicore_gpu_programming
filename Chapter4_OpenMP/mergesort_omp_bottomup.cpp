/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Parallel bottom-up mergesort, using OpenMP
 To build use  : g++ -fopenmp mergesort_omp_bottomup.cpp -o mergesort_omp_bottomup
 ============================================================================
 */
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <omp.h>
using namespace std;

#define MAXVALUE 10000
#define min(A,B) ((A<B) ? A : B)
#define max(A,B) ((A>B) ? A : B)
//------------------------------------
void numberGen (int N, int max, int *store)
{
  int i;
  srand (time (0));
  for (i = 0; i < N; i++)
    store[i] = rand () % max;
}

//------------------------------------
template < class T > void mergeList (T * src1, T * src2, int len1, int len2, T * dest)
{
  int idx1 = 0, idx2 = 0;
  int loc = 0;                  // starting point in dest array
  while (idx1 < len1 && idx2 < len2)
    {
      if (src1[idx1] <= src2[idx2])
        {
          dest[loc] = src1[idx1];
          idx1++;
        }
      else
        {
          dest[loc] = src2[idx2];
          idx2++;
        }
      loc++;
    }

  // copy the rest
  for (int i = idx1; i < len1; i++)
    dest[loc++] = src1[i];

  for (int i = idx2; i < len2; i++)
    dest[loc++] = src2[i];
}

//--------------------------------------------------------
template < class T > void mergesort (T * data, int N)
{
  // allocate temporary array
  T *temp = new T[N];
  // pointers to easily switch between the two arrays
  T *repo1, *repo2, *aux;

  repo1 = data;
  repo2 = temp;

  // loop for group size growing exponentially from 1 element to floor(lgN)
  for (int grpSize = 1; grpSize < N; grpSize <<= 1)
    {
#pragma omp parallel for
      for (int stIdx = 0; stIdx < N; stIdx += 2 * grpSize)
        {
          int nextIdx = stIdx + grpSize;
          int secondGrpSize = min (max (0, N - nextIdx), grpSize);

          // check to see if there are enough data for a second group to merge with       
          if (secondGrpSize == 0)
            {
              // if there is no second part, just copy the first part to repo2 for use in the next iteration
              for (int i = 0; i < N - stIdx; i++)
                repo2[stIdx + i] = repo1[stIdx + i];
            }
          else
            {
              mergeList (repo1 + stIdx, repo1 + nextIdx, grpSize, secondGrpSize, repo2 + stIdx);
            }
        }

      // switch pointers
      aux = repo1;
      repo1 = repo2;
      repo2 = aux;
    }


  // move data back to the original array  
  if (repo1 != data)
    memcpy (data, temp, sizeof (T) * N);

  delete[]temp;
}

//--------------------------------------------------------
int main (int argc, char *argv[])
{
  if (argc < 2)
    {
      cout << "Use : " << argv[0] << " numData\n";
      exit (1);
    }

  int N = atoi (argv[1]);
  int *data = new int[N];
  numberGen (N, MAXVALUE, data);

  double t = omp_get_wtime ();
  mergesort (data, N);
  // printout sorting time in sec
  cout << omp_get_wtime () - t << endl;

  // Sanity check
  for (int i = 0; i < N - 1; i++)
    if (data[i + 1] < data[i])
      cout << "ERROR!\n";

  delete[]data;
  return 0;
}
