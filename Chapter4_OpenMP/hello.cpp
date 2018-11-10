/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp hello.cpp -o hello
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace std;

int main (int argc, char **argv)
{
  int numThr = atoi (argv[1]);
#pragma omp parallel num_threads(numThr)
  cout << "Hello from thread " << omp_get_thread_num () << endl;

  return 0;
}
