/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake raceCond.pro; make
 ============================================================================
 */
#include <QThread>
#include <iostream>
#include <unistd.h>

using namespace std;

int counter = 0;

class MyThread:public QThread {
private:
  int ID;
  int runs;
public:
  MyThread (int i, int r):ID(i), runs(r) {}
  void run ()  {
    cout << "Thread " << ID << " is running\n";
    for (int j = 0; j < runs; j++)
      {
	usleep (rand () % 3);
	counter++;
      }
  }
};

int main (int argc, char *argv[])
{
  srand (time (0));
  int N = atoi(argv[1]);
  int runs = atoi(argv[2]);
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
