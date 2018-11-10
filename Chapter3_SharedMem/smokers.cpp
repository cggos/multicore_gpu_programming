/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake smokers.pro; make
 ============================================================================
 */
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QMutexLocker>
#include <iostream>
#include <stdlib.h>

using namespace std;
#define TOBACCO_PAPER 0  
#define TOBACCO_MATHCES 1
#define PAPER_MATHCES 2
#define MAXSLEEP 1000
const char *msg[]={"having matches", "having paper", "having tobacco"};
//***************************************************
class Monitor
{
  private:
     QMutex l;
     QWaitCondition w, finish;
     int newingred;   
     int exitf;
  public:
     Monitor();
    // return 0 if OK. Otherwise it means termination
     int canSmoke(int);
     void newIngredient(int );
     void finishedSmoking();
     void finishSim();
};
//--------------------------------------------
Monitor::Monitor(): newingred(-1), exitf(0) {}
//--------------------------------------------
void Monitor::newIngredient(int newi)
{
  QMutexLocker ml(&l);  
  newingred = newi;
  w.wakeAll();
  finish.wait(&l);
}
//--------------------------------------------
int Monitor::canSmoke(int missing)
{
  QMutexLocker ml(&l);  
  while(newingred != missing && ! exitf)
    w.wait(&l);
  return exitf;
}
//--------------------------------------------
void Monitor::finishedSmoking()
{
  newingred = -1;
  QMutexLocker ml(&l);  
  finish.wakeOne();
}
//--------------------------------------------
void Monitor::finishSim()
{
  exitf=1;
  w.wakeAll();
}
//***************************************************
class Smoker : public QThread
{
  private:
    int missing_ingred;
    Monitor *m;
    int total;
  public:
    Smoker(int, Monitor *);
    void run();
};
//--------------------------------------------
Smoker::Smoker(int missing, Monitor *mon) : missing_ingred(missing), m(mon), total(0){}
//--------------------------------------------
void Smoker::run()
{
  while((m->canSmoke(missing_ingred)) ==0)
  {
     total++;
     cout << "Smoker " << msg[missing_ingred] << " is smoking\n";
     msleep(rand() % MAXSLEEP);
     m->finishedSmoking();
  }
// cout << "Smoker " << msg[missing_ingred] << " smoked a total of " << total << "\n";
}
//***************************************************
class Agent : public QThread
{
  private:
    int runs;
    Monitor *m;
  public:
    Agent(int, Monitor *);
    void run(); 
};
//--------------------------------------------
Agent::Agent(int r, Monitor *mon) : runs(r), m(mon){}
//--------------------------------------------
void Agent::run()
{
   for(int i=0;i<runs; i++)
   {
      int ingreds = rand() % 3;
      m->newIngredient(ingreds);      
   }
  m->finishSim();      
}
//***************************************************
int main(int argc, char **argv)
{
   Monitor m;
   Smoker *s[3];
   for(int i=0;i<3;i++)
   { 
     s[i] = new Smoker(i, &m);
     s[i]->start();
   }
   Agent a(atoi(argv[1]), &m);
   a.start();
   
   a.wait();
   for(int i=0;i<3;i++)
     s[i]->wait();
   
   return EXIT_SUCCESS;
}
