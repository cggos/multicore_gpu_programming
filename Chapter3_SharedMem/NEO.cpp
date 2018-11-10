/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake NEO.pro; make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <QRunnable>
#include <QThreadPool>
#include <QtConcurrentFilter>
#include <QVector>
#include <QTime>

using namespace std;

struct NEO
{
   double radius;
   double mass;
   int groupNum;
   char *name;
};
//--------------------------------------------------------
bool filtFunc(const NEO &x)
{
  return  x.radius >= 10;  
}
//--------------------------------------------------------
void reduFunc(NEO  &x, const NEO  &y)
{
  x.mass += y.mass;
  x.groupNum += y.groupNum;
}
//--------------------------------------------------------
int main (int argc, char *argv[])
{
  QVector<NEO> data;
  for(int i=0;i<10;i++)
  {
    NEO aux;
    aux.radius = i*10;
   aux.mass = i*100;
   aux.groupNum = 1;
    data.append(aux);
  }
  QTime t;
  t.start();
  NEO average = QtConcurrent::blockingFilteredReduced(data, filtFunc, reduFunc);
  cout << t.elapsed() << endl;
  cout << "Average " << average.mass / average.groupNum << endl;

  for(int i=0;i<10;i++)
  {
     cout << data[i].radius << " " << data[i].mass << endl;
  }

  return 0;
}
