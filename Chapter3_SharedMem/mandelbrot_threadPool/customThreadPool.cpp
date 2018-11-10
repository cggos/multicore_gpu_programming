/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake ; make
 ============================================================================
 */
#include <QThread>
#include <QWaitCondition>
#include <QMutex>
#include <QImage>
#include <QMutexLocker>
#include <set>
#include "customThreadPool.h"

using namespace std;

const int CustomThreadPool::BUFFERSIZE = 100;
const int CustomThreadPool::NUMTHREADS = 16;
//--------------------------------------
CustomThreadPool *CustomThread::tp;
//--------------------------------------

void CustomThread::run() {
    ComputationalTask *task = tp->get();
    while (task != NULL) {
        task->compute();
        tp->complete(task->getTaskID());
        task = tp->get();
    }
}
//************************************************************

CustomThreadPool::CustomThreadPool(int n, int numThr) {
    N = n;
    buffer = new ComputationalTask*[n];
    in = out = count = 0;
    nextTaskID = 0;
    maxThreads = numThr;
    t = new CustomThread*[maxThreads];
    CustomThread::tp = this;
    for (int i = 0; i < maxThreads; i++) {
        t[i] = new CustomThread();
        t[i]->start();
    }
}
//--------------------------------------

CustomThreadPool::~CustomThreadPool() {
    for (int i = 0; i < maxThreads; i++)
        this->schedule(NULL);

    for (int i = 0; i < maxThreads; i++) {
        this->t[i]->wait();
        delete t[i];
    }
    delete []t;
    delete []buffer;
}
//--------------------------------------

unsigned int CustomThreadPool::schedule(ComputationalTask *ct) {
    QMutexLocker ml(&l);
    while (count == N)
        full.wait(&l);
    buffer[in] = ct;
    in = (in + 1) % N;
    count++;

    if (ct != NULL) // check it is not the termination task
    {
        ct->setTaskID(nextTaskID);
        nextTaskID++;
    }

    empty.wakeOne();

    return (nextTaskID - 1);
}
//--------------------------------------

ComputationalTask *CustomThreadPool::get() {
    QMutexLocker ml(&l);
    while (count == 0)
        empty.wait(&l);

    ComputationalTask *temp = buffer[out];
    out = (out + 1) % N;
    count--;

    full.wakeOne();
    return temp;
}
//--------------------------------------

void CustomThreadPool::complete(unsigned int id) {
    QMutexLocker ml(&l);
    finished.insert(id);
    notDone.wakeAll();
}
//--------------------------------------

void CustomThreadPool::waitTillDone(unsigned int id) {
    QMutexLocker ml(&l);
    while (finished.find(id) == finished.end())
        notDone.wait(&l);
    finished.erase(id);
}
