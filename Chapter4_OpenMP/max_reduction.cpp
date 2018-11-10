/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp max_reduction.cpp -o max_reduction
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main (int argc, char **argv)
{
  int M = 100;
  int data[M];

  int maxElem = data[0];
#pragma omp parallel for reduction(max : maxElem)
  for (int i = 1; i < sizeof (data) / sizeof (int); i++)
    if (maxElem < data[i])
      maxElem = data[i];

  cout << "Maximum is : " << maxElem << endl;

  return 0;
}
