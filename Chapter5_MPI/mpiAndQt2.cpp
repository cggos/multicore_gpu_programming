/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : qmake mpiAndQt2.pro; make
 ============================================================================
 */
#include<mpi.h>
#include<iostream>
#include<unistd.h>
#include <QThread>

using namespace std;
//---------------------------------------
class MyThread:public QThread
{
private:
  int ID, rank;
public:
    MyThread (int i, int r):ID (i), rank (r) {}
  void run ()
  {
    cout << "Thread " << ID << " is running on process " << rank << "\n";
  }
};

//---------------------------------------
int main (int argc, char **argv)
{
  int support;
  MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &support);

  int rank, N;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &N);

  int numThreads = sysconf (_SC_NPROCESSORS_ONLN);
  MyThread *x[numThreads];
  for (int i = 0; i < numThreads; i++)
    {
      x[i] = new MyThread (i, rank);
      x[i]->start ();
    }

  for (int i = 0; i < numThreads; i++)
    x[i]->wait ();

  MPI_Finalize ();
  return 0;
}
