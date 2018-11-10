/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake writersFav.pro; make
 ============================================================================
 */
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QWaitCondition>
#include <iostream>
#include <stdlib.h>

using namespace std;

const int NUMOPER=3;
//*************************************
class Monitor
{
 private:
    QMutex l;
    QWaitCondition wq;  // for blocking writers
    QWaitCondition rq;  // for blocking readers
    int readersIn;  // how many readers in their critical section
    bool writerIn;  // set if a write is in its critical section
    int writersWaiting; // how many writers are waiting to enter
 public:
   Monitor() : readersIn(0), writerIn(0), writersWaiting(0) {}
   void canRead();
   void finishedReading();
   void canWrite();
   void finishedWriting();
};
//*************************************
class Reader: public QThread
{
 private:
   int ID;
   Monitor *coord;
 public:
   Reader(int i, Monitor *c) : ID(i), coord(c) {}
   void run();
};
//*************************************
void Reader::run()
{
  for(int i=0;i<NUMOPER;i++)
    {
       coord->canRead();
       cout << "Reader " << ID << " read oper. #" << i << endl;
       sleep(rand()%4+1);

       coord->finishedReading();
    }
}
//*************************************
class Writer: public QThread
{
 private:
   int ID;
   Monitor *coord;
 public:
   Writer(int i, Monitor *c) : ID(i), coord(c) {}
   void run();
};
//*************************************
void Writer::run()
{
  for(int i=0;i<NUMOPER;i++)
    {
       coord->canWrite();
       cout << "Writer " << ID << " write oper. #" << i << endl;
       sleep(rand()%4+1);
       coord->finishedWriting();
    }
}
//*************************************
void Monitor::canRead()
{
  QMutexLocker ml(&l);
  while(writerIn==true || writersWaiting>0)
     rq.wait(&l);

  readersIn++;  
}
//*************************************
void Monitor::canWrite()
{
  QMutexLocker ml(&l);
  while(writerIn==true ||  readersIn>0)
  {
     writersWaiting++;
     wq.wait(&l);
     writersWaiting--;
  }

  writerIn=true;
}
//*************************************
void Monitor::finishedReading()
{
  QMutexLocker ml(&l);
  readersIn--;
  if(readersIn == 0)
     wq.wakeOne();
}
//*************************************
void Monitor::finishedWriting()
{
  QMutexLocker ml(&l);
  writerIn=false;
  if(writersWaiting>0)
      wq.wakeOne();
  else
      rq.wakeAll();
}
//*************************************
int main(int argc, char** argv)
{
  int numRead=atoi(argv[1]);
  int numWrite = atoi(argv[2]);
  Monitor m;
  Reader *r[numRead];
  Writer *w[numWrite];

  srand(clock());

  for(int i=0;i<numRead;i++)
    {
       r[i] = new Reader(i, &m);
       r[i]->start();
    }
  for(int i=0;i<numWrite;i++)
    {
       w[i] = new Writer(i, &m);
       w[i]->start();
    }


  for(int i=0;i<numRead;i++)
       r[i]->wait();

  for(int i=0;i<numWrite;i++)
       w[i]->wait();

  return 0;
}
