/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake qtconcurProdCons.pro; make
 ============================================================================
 */
#include <QtConcurrentRun>
#include <QFuture>
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <iostream>
#include <unistd.h>

using namespace std;

const int BUFFSIZE = 100;

template<typename T>
class Producer {
private:
    int ID;
    static QSemaphore *slotsAvail;
    static QSemaphore *resAvail;
    static QMutex l1;
    static QSemaphore numProducts;
    static T *buffer;
    static int in;
public:
    static T(*produce)();
    static void initClass(int numP, QSemaphore *s, QSemaphore *a, T *b, T(*prod)());

    Producer<T>(int i) : ID(i) {
    };
    void run();
};
//---------------------------------------
template<typename T> QSemaphore Producer<T>::numProducts;
template<typename T> QSemaphore * Producer<T>::slotsAvail;
template<typename T> QSemaphore * Producer<T>::resAvail;
template<typename T> QMutex Producer<T>::l1;
template<typename T> T * Producer<T>::buffer;
template<typename T> int Producer<T>::in = 0;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T> void Producer<T>::initClass(int numP, QSemaphore *s, QSemaphore *a, T *b, T(*prod)()) {
    numProducts.release(numP);
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    produce = prod;
}
//---------------------------------------  

template<typename T>
void Producer<T>::run() {
    while (numProducts.tryAcquire()) {
        T item = (*produce)();
        slotsAvail->acquire(); // wait for an empty slot in the buffer
        l1.lock();
        int tmpIn = in;
        in = (in + 1) % BUFFSIZE; // update the in index safely
        l1.unlock();
        buffer[tmpIn] = item; // store the item
        resAvail->release(); // signal resource availability
    }
}
//---------------------------------------  

template<typename T>
class Consumer {
private:
    int ID;
    static QSemaphore *slotsAvail;
    static QSemaphore *resAvail;
    static QMutex l2;
    static T *buffer;
    static int out;
    static QSemaphore numProducts;
public:
    static void (*consume)(T i);
    static void initClass(int numP, QSemaphore *s, QSemaphore *a, T *b, void (*cons)(T));

    Consumer<T>(int i) : ID(i) {
    };
    void run();
};
//---------------------------------------

template<typename T> QSemaphore Consumer<T>::numProducts;
template<typename T> QSemaphore * Consumer<T>::slotsAvail;
template<typename T> QSemaphore * Consumer<T>::resAvail;
template<typename T> QMutex Consumer<T>::l2;
template<typename T> T * Consumer<T>::buffer;
template<typename T> int Consumer<T>::out = 0;
template<> void (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> void Consumer<T>::initClass(int numP, QSemaphore *s, QSemaphore *a, T *b, void (*cons)(T)) {
    numProducts.release(numP);
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    consume = cons;
}
//---------------------------------------

template<typename T> void Consumer<T>::run() {
    while (numProducts.tryAcquire()) {
        resAvail->acquire(); // wait for an available item
        l2.lock();
        int tmpOut = out;
        out = (out + 1) % BUFFSIZE; // update the out index
        l2.unlock();
        T item = buffer[tmpOut]; // take the item out
        slotsAvail->release(); // signal for a new empty slot 
        (*consume)(item);
    }
}
//---------------------------------------

int produce() {
    // to be implemented
    return 1;
}
//---------------------------------------

void consume(int i) {
    // to be implemented
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #producers #consumers #iterations\n";
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int numP = atoi(argv[3]);
    int *buffer = new int[BUFFSIZE];
    QSemaphore avail, buffSlots(BUFFSIZE);

    Producer<int>::initClass(numP, &buffSlots, &avail, buffer, &produce);
    Consumer<int>::initClass(numP, &buffSlots, &avail, buffer, &consume);

    if(N + M > QThreadPool::globalInstance() -> maxThreadCount())
        QThreadPool::globalInstance() -> setMaxThreadCount (N+M);
    
    QFuture<void> prodF[N];
    QFuture<void> consF[M];
    Producer<int> *p[N];
    Consumer<int> *c[M];

    for (int i = 0; i < N; i++) {
        p[i] = new Producer<int>(i);
        prodF[i] = QtConcurrent::run(*p[i], &Producer<int>::run);
    }

    for (int i = 0; i < M; i++) {
        c[i] = new Consumer<int>(i);
        consF[i] = QtConcurrent::run(*c[i], &Consumer<int>::run);
    }

    for (int i = 0; i < N; i++)
        prodF[i].waitForFinished();

    for (int i = 0; i < M; i++)
        consF[i].waitForFinished();


    delete [] buffer;
    return 0;
}
