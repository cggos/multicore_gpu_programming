/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake readWriteFair.pro; make
 ============================================================================
 */
#include <QThread>
#include <QMutex>
#include <QMutexLocker>
#include <QWaitCondition>
#include <iostream>
#include <stdlib.h>

using namespace std;

const int QUESIZE = 100;
const int NUMOPER = 3;
//*************************************

class Monitor
{
private:
  QMutex l;
  QWaitCondition c[QUESIZE];    // a different condition for each waiting thread
  bool writeflag[QUESIZE];      // what kind of threads wait?
  QWaitCondition quefull;       // used when queue of waiting threads becomes full
  int in, out, counter;
  int readersIn;                // how many readers in their critical section
  int writersIn;                // how many writers in their critical section (0 or 1)
public:

    Monitor ():in (0), out (0), counter (0), readersIn (0), writersIn (0)
  {
  }
  void canRead ();
  void finishedReading ();
  void canWrite ();
  void finishedWriting ();
};

//*************************************

class Reader:public QThread
{
private:
  int ID;
  Monitor *coord;
public:

    Reader (int i, Monitor * c):ID (i), coord (c)
  {
  }
  void run ();
};

//*************************************

void Reader::run ()
{
  for (int i = 0; i < NUMOPER; i++)
    {
      coord->canRead ();
      cout << "Reader " << ID << " read oper. #" << i << endl;
      sleep (rand () % 4 + 1);

      coord->finishedReading ();
    }
}

//*************************************

class Writer:public QThread
{
private:
  int ID;
  Monitor *coord;
  int delay;
public:

  Writer (int i, Monitor * c):ID (i), coord (c)
  {
  }
  void run ();
};

//*************************************

void Writer::run ()
{
  for (int i = 0; i < NUMOPER; i++)
    {
      coord->canWrite ();
      cout << "Writer " << ID << " write oper. #" << i << endl;
      sleep (rand () % 4 + 1);
      coord->finishedWriting ();
    }
}

//*************************************

void Monitor::canRead ()
{
  QMutexLocker ml (&l);
  while (counter == QUESIZE)
    quefull.wait (&l);

  if (counter > 0 || writersIn)
    {
      int temp = in;
      writeflag[in] = false;
      in = (in + 1) % QUESIZE;
      counter++;
      c[temp].wait (&l);
    }
  readersIn++;
}

//*************************************

void Monitor::canWrite ()
{
  QMutexLocker ml (&l);
  while (counter == QUESIZE)
    quefull.wait (&l);

  if (counter > 0 || writersIn > 0 || readersIn > 0)
    {
      int temp = in;
      writeflag[in] = true;
      in = (in + 1) % QUESIZE;
      counter++;
      c[temp].wait (&l);
    }
  writersIn++;
}

//*************************************

void Monitor::finishedReading ()
{
  QMutexLocker ml (&l);
  readersIn--;
  if (readersIn == 0 && counter > 0)
    {
      c[out].wakeOne ();        // it must be a writer that is being woken up
      out = (out + 1) % QUESIZE;
      counter--;
      quefull.wakeOne ();
    }
}

//*************************************

void Monitor::finishedWriting ()
{
  QMutexLocker ml (&l);
  writersIn--;
  if (counter > 0)
    {
      if (!writeflag[out])
        {
          while (counter > 0 && !writeflag[out])        // start next readers
            {
              c[out].wakeOne ();
              out = (out + 1) % QUESIZE;
              counter--;
            }
        }
      else                      // next writer
        {
          c[out].wakeOne ();
          out = (out + 1) % QUESIZE;
          counter--;
        }
      quefull.wakeAll ();
    }
}

//*************************************

int main (int argc, char **argv)
{
  int numRead = atoi (argv[1]);
  int numWrite = atoi (argv[2]);
  Monitor m;
  Reader *r[numRead];
  Writer *w[numWrite];

  srand (clock ());

  for (int i = 0; i < numRead; i++)
    {
      r[i] = new Reader (i, &m);
      r[i]->start ();
    }
  for (int i = 0; i < numWrite; i++)
    {
      w[i] = new Writer (i, &m);
      w[i]->start ();
    }


  for (int i = 0; i < numRead; i++)
    r[i]->wait ();

  for (int i = 0; i < numWrite; i++)
    w[i]->wait ();

  return 0;
}
