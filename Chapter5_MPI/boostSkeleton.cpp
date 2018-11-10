/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC boostSkeleton.cpp -o boostSkeleton -lboost_mpi -lboost_serialization
 ============================================================================
 */
#include <boost/mpi.hpp>
#include <string>
#include <iostream>

using namespace std;
using namespace boost;

const int TAGSKELETON = 0;
const int TAGWORKITEM = 1;
//============================================================
struct WorkItem
{
public:
  int param1;
  double param2;
  string param3;

    template < class Arch > void serialize (Arch & r, int version)
  {
    r & param1;
    r & param2;
    r & param3;
  }
};

//============================================================
int main (int argc, char **argv)
{
  mpi::environment env (argc, argv);
  mpi::communicator world;

  WorkItem item;
  item.param3 = "work";
  int rank = world.rank ();
  int N = world.size ();
  int numWork = atoi (argv[1]);

  if (rank == 0)
    {
      for (int i = 1; i < N; i++)
        world.send (i, TAGSKELETON, mpi::skeleton (item));

      int destID = 1;
      for (int i = 0; i < numWork; i++)
        {
          item.param1 = i;
          world.send (destID, TAGWORKITEM, mpi::get_content (item));
          destID = (destID == N - 1) ? 1 : destID + 1;
        }
      item.param3 = "done";
      for (int i = 1; i < N; i++)
        world.send (i, 1, mpi::get_content (item));
    }
  else
    {
      world.recv (0, TAGSKELETON, mpi::skeleton (item));
      world.recv (0, TAGWORKITEM, mpi::get_content (item));
      while (item.param3 != "done")
        {
          cout << "Worker " << rank << " got " << item.param1 << endl;
          world.recv (0, TAGWORKITEM, mpi::get_content (item));
        }
    }

  return 0;
}
