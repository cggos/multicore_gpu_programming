#ifndef WORKQUEUE_H
#define WORKQUEUE_H

#include <QMutex>
#include <queue>
#include "mandelregion.h"

using namespace std;

class WorkQueue
{
private:
  QMutex l;
  deque<MandelRegion *> queue;
  
public:
  void append(MandelRegion*);
  MandelRegion* extract();
  int size();
};
#endif