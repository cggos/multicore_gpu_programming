/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC boostReduction.cpp -o boostReduction -lboost_mpi -lboost_serialization
 ============================================================================
 */
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <iostream>

using namespace std;
using namespace boost;
//============================================================
class CenterOfMass:public std::binary_function < CenterOfMass, CenterOfMass, CenterOfMass >
{
public:
  double x, y, z, mass;

  CenterOfMass (double a, double b, double c, double m):x (a), y (b), z (c), mass (m) {}

  CenterOfMass () {}

  template < class Archive > void serialize (Archive & ar, const unsigned int version);
  const CenterOfMass & operator () (const CenterOfMass & o1, const CenterOfMass & o2) const;
};

//---------------------------------------------------
template < class Archive > void CenterOfMass::serialize (Archive & ar, const unsigned int version)
{
  ar & x;
  ar & y;
  ar & z;
  ar & mass;
}

//---------------------------------------------------    
const CenterOfMass & CenterOfMass::operator () (const CenterOfMass & o1, const CenterOfMass & o2)
     const
     {
       CenterOfMass *res = new CenterOfMass ();
       res->x = o1.x * o1.mass + o2.x * o2.mass;
       res->y = o1.y * o1.mass + o2.y * o2.mass;
       res->z = o1.z * o1.mass + o2.z * o2.mass;
       double M = o1.mass + o2.mass;
       res->x /= M;
       res->y /= M;
       res->z /= M;
       res->mass = M;
       return *res;
     }

//============================================================  
ostream & operator<< (ostream & out, const CenterOfMass & obj)
{
  out << obj.mass << " at (" << obj.x << ", " << obj.y << ", " << obj.z << ") " << endl;;
}

//============================================================
int main (int argc, char **argv)
{
  mpi::environment env (argc, argv);
  mpi::communicator world;

  int rank = world.rank ();
  
  CenterOfMass m1 (1.0 * rank, 1.0 * rank, 1.0 * rank, 1.0 * rank + 1);
  CenterOfMass m2;
  
  reduce (world, m1, m2, CenterOfMass (), 0);
  if (world.rank () == 0)
    cout << m2 << endl;
  return 0;
}
