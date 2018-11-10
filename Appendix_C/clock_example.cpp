#include <time.h>
#include <iostream>

using namespace std;

long time0_nsec = 0;

double time0_sec = 0;

//***********************************************
double hrclock_sec ()
{
  timespec ts;
  clock_gettime (CLOCK_REALTIME, &ts);
  double aux = ts.tv_sec + ts.tv_nsec / 1000000000.0;
  return aux - time0_sec;
}

//***********************************************

long hrclock_nsec ()
{
  timespec ts;
  clock_gettime (CLOCK_REALTIME, &ts);
  long aux = ts.tv_sec * 1000000000 + ts.tv_nsec;
  aux -= time0_nsec;
  return aux;
}

//***********************************************

int main ()
{
  time0_nsec = hrclock_nsec ();

  time0_sec = hrclock_sec ();
  sleep (1);
  cout << hrclock_nsec () << endl;
  cout << hrclock_sec () << endl;
  return 0;
};
