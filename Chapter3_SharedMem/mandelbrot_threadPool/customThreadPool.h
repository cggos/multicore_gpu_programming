#include <QThread>
#include <QWaitCondition>
#include <QMutex>
#include <QImage>
#include <QMutexLocker>
#include <set>

using namespace std;

class CustomThreadPool;

//************************************************************
class ComputationalTask {
   private:
    unsigned int taskID;
   public:
    virtual void compute() = 0;
    void setTaskID(int id) { taskID = id;}    // used only by CustomThreadPool
    unsigned int getTaskID() {return taskID;} // used only by CustomThreadPool
};

//************************************************************
class CustomThread : public QThread
{
  public:
    static CustomThreadPool *tp;
    void run();
};
//************************************************************
class CustomThreadPool
{
private:
    static const int BUFFERSIZE;
    static const int NUMTHREADS;
    QWaitCondition notDone;   // for blocking while a thread is not finished
    QWaitCondition empty;     // for blocking pool-threads if buffer is empty
    QWaitCondition full;     
    QMutex l;
    ComputationalTask **buffer;  // pointer to array of pointers
    int in, out, count, N, maxThreads;
    unsigned int nextTaskID;    // used to enumerate the assigned tasks
    set<unsigned int> finished; // keeps the task IDs of finished tasks.
                                // IDs are removed from the set, once the isDone
                                // method is called for them
    CustomThread **t;
public:
    CustomThreadPool(int n = BUFFERSIZE, int nt = NUMTHREADS);
    ~CustomThreadPool();

    ComputationalTask *get();    // to be called by the pool-threads
    void complete(unsigned int); // to be called by the pool-threads

    unsigned int schedule(ComputationalTask *);
    void waitTillDone(unsigned int); // to be called by the task generator
};

