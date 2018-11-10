/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ nested_flowDep.cpp -o nested_flowDep
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>

using namespace std;

int main (int argc, char **argv)
{
  int N = 5, M = 5;
  double **data = new double *[N];
  for (int i = 0; i < M; i++)
    data[i] = new double[M];

  // init with sample values
  for (int i = 0; i < N; i++)
    data[i][0] = 1;
  
  for (int j = 0; j < M; j++)
    data[0][j] = 1;

  // compute
  for (int i = 1; i < N; i++)
    for (int j = 1; j < M; j++)
      data[i][j] = data[i - 1][j] + data[i][j - 1] + data[i - 1][j - 1];

  //output
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
        cout << data[i][j] << " ";
      cout << endl;
    }

  return 0;
}
