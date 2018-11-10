/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp flush_test.cpp -o flush_test
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>

using namespace std;

int main (int argc, char **argv)
{

  bool flag = false;
#pragma omp parallel sections default( none ) shared( flag, cout )
  {
#pragma omp section
    {
      // wait for signal
      while (flag == false)
      {
#pragma omp flush ( flag )	
      }
      // do something 
      cout << "First section\n";
    }
#pragma omp section
    {
      // do something first
      cout << "Second section\n";
      sleep (1);
      // signal other section
      flag = true;
#pragma omp flush ( flag )

    }
  }

  return 0;
}
