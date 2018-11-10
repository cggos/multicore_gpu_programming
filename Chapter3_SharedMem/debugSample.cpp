/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake debugSample.pro; make
 ============================================================================
 */
#include <QThread>
#include <QMutex>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <time.h>

using namespace std;

#define DEBUG

//***********************************************
double time0 = 0;
QMutex l;
double hrclock ()
{
  timespec ts;
  clock_gettime (CLOCK_REALTIME, &ts);
  double aux = ts.tv_sec + ts.tv_nsec / 1000000000.0;
  return aux - time0;
}

//***********************************************
void debugMsg (string msg, double timestamp)
{
  l.lock ();
  cerr << timestamp << " " << msg << endl;
  l.unlock ();
}

//***********************************************
int counter = 0;

class MyThread:public QThread
{
private:
  int ID;
  int runs;
public:
    MyThread (int i, int r):ID (i), runs (r)
  {
  }
  void run ()
  {
    cout << "Thread " << ID << " is running\n";
    for (int j = 0; j < runs; j++)
      {
#ifdef DEBUG
        ostringstream ss;
        ss << "Thread #" << ID << " counter=" << counter;
        debugMsg (ss.str (), hrclock ());
#endif
        usleep (rand () % 3);
        counter++;
      }
  }
};

int main (int argc, char *argv[])
{
#ifdef DEBUG
  time0 = hrclock ();
#endif

  srand (time (0));
  int N = atoi (argv[1]);
  int runs = atoi (argv[2]);
  MyThread *t[N];
  for (int i = 0; i < N; i++)
    {
      t[i] = new MyThread (i, runs);
      t[i]->start ();
    }


  for (int i = 0; i < N; i++)
    t[i]->wait ();

  cout << counter << endl;
  return 0;
}
