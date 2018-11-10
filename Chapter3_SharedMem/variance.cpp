/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake variance.pro; make
 ============================================================================
 */
// G. Barlas, Oct. 2013
#include <QtConcurrentMap>
#include <QList>
#include <stdlib.h>
#include <iostream>

using namespace std;
//******************************************
long powX2 (const long &x)
{
  return x * x;
}

//******************************************
void sumFunc (long &x, const long &y)
{
  x += y;
}

//******************************************
int main (int argc, char *argv[])
{
  srand (time (0));
  int N = atoi (argv[1]);
  QList < long >data;
  for (int i = 0; i < N; i++)
    data.append ((rand () % 2000000) - 1000000);

  // because during reduction only one thread is working at a time, average is calculated sequentially
  long sum = 0;
  for (int i = 0; i < N; i++)
    sum += data[i];
  double mean = sum * 1.0 / N;

  long res = QtConcurrent::blockingMappedReduced (data, powX2, sumFunc);
  double var = res * 1.0 / N - mean * mean;
  cout << "Average: " << mean << " Variance " << var << endl;

  return 0;
}
