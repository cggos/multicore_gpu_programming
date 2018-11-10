/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : g++ -std=c++11 cpp11Thread.cpp -o cpp11Thread
 ============================================================================
 */
#include <thread>
#include <iostream>

using namespace std;

void hello()
{
   cout << "Hello from the child thread\n"; 
}

int main()
{
  thread t(hello);
  cout << "Hello from the parent thread\n"; 
  t.join();
  return 0;
}