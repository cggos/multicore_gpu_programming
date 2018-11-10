/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc hello.cu -o hello
 ============================================================================
 */
#include <iostream>
#include <thrust/version.h>

using namespace std;

int main()
{ 
  cout << "Hello World from Thrust v " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl;
  return 0;   
}