/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake qtThreads.pro; make
 ============================================================================
 */
#include <QThread>
#include <iostream>

using namespace std;

class MyThread:public QThread
{
private:
  int ID;
public:
  MyThread (int i):ID (i) {}
  void run ()  {
    cout << "Thread " << ID << " is running\n";
  }
};

int main (int argc, char *argv[])
{
  int N = atoi (argv[1]);
  MyThread *x[N];
  for (int i = 0; i < N; i++)
    {
      x[i] = new MyThread (i);
      x[i]->start ();
    }

  for (int i = 0; i < N; i++)
    x[i]->wait ();
  return 0;
}
