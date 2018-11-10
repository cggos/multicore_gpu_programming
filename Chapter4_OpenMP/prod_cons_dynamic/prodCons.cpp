/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : One producer, multiple -dynamic- consumers solution
 To build use  : qmake; make
 ============================================================================
 */
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <QSemaphore>
#include <QMutex>

using namespace std;

const int BUFFSIZE = 10;
const double LOWERLIMIT = 0;
const double UPPERLIMIT = 10;

const int NUMCONSUMERS = 2;
//--------------------------------------------
typedef struct Slice
{
  double start;
  double end;
  int divisions;
} Slice;
//--------------------------------------------
double func (double x)
{
  return fabs (sin (x));
}

//--------------------------------------------
void integrCalc (Slice * buffer, QSemaphore &buffSlots, QSemaphore &avail, QMutex &l, int &out, QMutex &resLock, double &res)
{
  while (1)
    {
      avail.acquire ();        // wait for an available item
      l.lock ();
      int tmpOut = out;
      out = (out + 1) % BUFFSIZE;       // update the out index
      l.unlock ();

      // take the item out
      double st = buffer[tmpOut].start;
      double en = buffer[tmpOut].end;
      double div = buffer[tmpOut].divisions;

      buffSlots.release ();    // signal for a new empty slot 

      if (div == 0)
        break;                  // exit

      //calculate area  
      double localRes = 0;
      double step = (en - st) / div;
      double x;
      x = st;
      localRes = func (st) + func (en);
      localRes /= 2;
      for (int i = 1; i < div; i++)
        {
          x += step;
          localRes += func (x);
        }
      localRes *= step;

      // add it to result
      resLock.lock ();
      res += localRes;
      resLock.unlock ();
    }
}

//--------------------------------------------
int main (int argc, char **argv)
{
  if (argc == 1)
    {
      cerr << "Usage " << argv[0] << " #threads #jobs\n";
      exit (1);
    }
  int N = atoi (argv[1]);
  int J = atoi (argv[2]);

  Slice *buffer = new Slice[BUFFSIZE];
  int in = 0, out = 0;
  QSemaphore avail, buffSlots (BUFFSIZE);
  QMutex l, integLock;
  double integral = 0;
#pragma omp parallel default(none) shared(buffer, in, out, avail, buffSlots, l, integLock, integral, J, N, cout) num_threads(N+1)
  {
// producer part    
#pragma omp single nowait
    {
      // consumer thread, responsible for handing out 'jobs'
      double divLen = (UPPERLIMIT - LOWERLIMIT) / J;
      double st, end = LOWERLIMIT;
      for (int i = 0; i < J; i++)
        {
          st = end;
          end += divLen;
          if (i == J - 1)
            end = UPPERLIMIT;

          buffSlots.acquire ();
          buffer[in].start = st;
          buffer[in].end = end;
          buffer[in].divisions = 1000;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }

      // put termination sentinels in buffer
      for (int i = 0; i < N; i++)
        {
          buffSlots.acquire ();
          buffer[in].divisions = 0;
          in = (in + 1) % BUFFSIZE;
          avail.release ();
        }
    }
    

// consumers' part
#pragma omp for schedule(static,1)
 for(int i=0;i<N;i++)
    {
       integrCalc (buffer, buffSlots, avail, l, out, integLock, integral);
    }
  }

  cout << "Result is : " << integral << endl;
  delete[]buffer;

  return 0;
}
