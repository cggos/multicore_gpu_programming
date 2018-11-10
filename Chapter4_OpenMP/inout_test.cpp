/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Example of task dependence. Requires an OpenMP 4.0 complaint compiler
 To build use  : g++ -fopenmp inout_test.cpp -o inout_test
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

int main (int argc, char **argv)
{
  int x = 1, y = 2;

#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task  shared(x) depend(in : x)
      {  // T1
        cout << "T1\n";
      }

#pragma omp task shared(x) depend(in : y)
      {  // T3
        cout << "T3\n";
      }

#pragma omp task  shared(x) depend(out : x, y)
      {  // T2
        sleep (1);
        cout << "T2\n";
      }

#pragma omp task shared(x) depend(in : x)
      {  // T4
        cout << "T4\n";
      }

#pragma omp task  shared(x) depend(in : x)
      {  // T5
	sleep(1);
        cout << "T5\n";
      }

#pragma omp task shared(x) depend(out : y)
      {  // T6
        cout << "T6\n";
      }
    } 
    
    
  }

return 0;
}
