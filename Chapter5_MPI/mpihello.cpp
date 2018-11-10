/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC mpihello.cpp -o mpihello -lboost_mpi
 ============================================================================
 */
#include<boost/mpi.hpp>
#include<iostream>
using namespace std;
int main (int argc, char **argv)
{
  boost::mpi::environment env (argc, argv);
  boost::mpi::communicator world;
  cout << "Hello from process " << world.rank () << " of " << world.size () << endl;
  return 0;
}
