/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ fork.cpp -o fork
 ============================================================================
 */
#include <cstdlib>
#include <iostream>
#include <unistd.h>

using namespace std;

int main (int argc, char **argv)
{
  pid_t child_id;

  child_id = fork ();
  if (child_id == 0)
    {
      cout << "This is the child process\n";
    }
  else
    {
      cout << "This is the parent process\n";
    }
  return 0;
}
