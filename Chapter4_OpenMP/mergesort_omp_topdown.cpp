/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Parallel top-down mergesort, using OpenMP
 To build use  : g++ -fopenmp mergesort_omp_topdown.cpp -o mergesort_omp_topdown
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

  memcpy (src1, dest, sizeof (T) * (len1 + len2));
}



//--------------------------------------------------------
// sort data array of N elements, using the aux array as temporary storage
template < class T > void mergesortRec (T * data, T * temp, int N)
{
  if (N < 2)
    return;
  else
    {
      int middle = N / 2;
#pragma omp task if(N>10000) mergeable
      {
        mergesortRec (data, temp, middle);
      }
#pragma omp task if(N>10000) mergeable
      {
        mergesortRec (data + middle, temp + middle, N - middle);
      }

#pragma omp taskwait

      mergeList (data, data + middle, middle, N - middle, temp);
    }
}

//--------------------------------------------------------
template < class T > void mergesort (T * data, int N)
{
  // allocate temporary array
  T *temp = new T[N];

#pragma omp parallel
  {

#pragma omp single
    {
      int middle = N / 2;
#pragma omp task
      {
        mergesortRec (data, temp, middle);
      }
#pragma omp task
      {
        mergesortRec (data + middle, temp + middle, N - middle);
      }

#pragma omp taskwait

      mergeList (data, data + middle, middle, N - middle, temp);
    }
  }

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
      {
//       cout << data[i] << " ";
        cout << "ERROR!\n";
        return 1;
      }

  delete[]data;
  return 0;
}
