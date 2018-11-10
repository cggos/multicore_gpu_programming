/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp integration_seq.cpp -o integration_seq
 ============================================================================
 */
// Sequential integration of testf
#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

//---------------------------------------
double testf (double x)
{
  return x * x + 2 * sin (x);
}

//---------------------------------------
double integrate (double st, double en, int div, double (*f) (double))
{
  double localRes = 0;
  double step = (en - st) / div;
  double x;
  x = st;
  localRes = f (st) + f (en);
  localRes /= 2;
  for (int i = 1; i < div; i++)
    {
      x += step;
      localRes += f (x);
    }
  localRes *= step;

  return localRes;
}

//---------------------------------------
int main (int argc, char *argv[])
{

  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " start end divisions\n";
      exit (1);
    }
  double start, end;
  int divisions;
  start = atof (argv[1]);
  end = atof (argv[2]);
  divisions = atoi (argv[3]);

  double finalRes = integrate (start, end, divisions, testf);

  cout << finalRes << endl;
  return 0;
}
