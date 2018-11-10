/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC boostnonblock.cpp -o boostnonblock -lboost_mpi -lboost_serialization
 ============================================================================
 */
#include<boost/mpi.hpp>
#include<boost/optional.hpp>
#include<vector>
#include<iostream>
using namespace std;
using namespace boost;
#define MESSTAG 0
int main (int argc, char **argv)
{
  mpi::environment env (argc, argv);
  mpi::communicator world;

  int rank = world.rank ();
  int N = world.size ();

  if (rank == 0)
    {
      string mess ("Hello World");
      vector < mpi::request > r; // a vector for the request objects
      vector < int >dest;        // a vector for the destination IDs
      for (int i = 1; i < N; i++)
        {
          r.push_back (world.isend (i, MESSTAG, mess));
          dest.push_back (i);
        }

      while (r.size() >0)
        {
          optional < pair < mpi::status, vector < mpi::request >::iterator > >res = mpi::test_any < vector < mpi::request >::iterator > (r.begin (), r.end ());
          if (res)
            {
              int idx = (res->second) - r.begin ();
              cout << "Message delivered to " << dest[idx] << endl;
              r.erase (res->second);   // remove completed operations from the vector
              dest.erase (dest.begin () + idx);
              
            }
        }
    }
  else   // worker code
    {
      string mess;
      mpi::request r = world.irecv (0, MESSTAG, mess);
      mpi::status s = r.wait ();        // block until communication is done
      cout << rank << " received " << mess << " - " << s.count < string >() << " item(s) received\n";
    }

  return 0;
}
