/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake qtThreadsSimple.pro; make
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
  MyThread(int id) : ID(id) {};
  void run ()  {
    cout << "Hello from the child thread " << ID << "\n";
  }
};

int main (int argc, char *argv[])
{
  MyThread t(0);
  t.start();
  cout << "Hello from the main thread\n";
  t.wait();
  return 0;
}
