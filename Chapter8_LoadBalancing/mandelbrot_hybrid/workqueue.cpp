#include "workqueue.h"

using namespace std;

void WorkQueue::append(MandelRegion* i)
{
  QMutexLocker ml(&l);
  queue.push_back(i);
}


MandelRegion* WorkQueue::extract()
{
  QMutexLocker ml(&l);
  if(size()==0) return NULL;

  MandelRegion* temp = queue.front();
  queue.pop_front();
  return temp;
}


int WorkQueue::size()
{
  return queue.size();
}
