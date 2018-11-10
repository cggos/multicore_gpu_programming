/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake monitor1ProdCons.pro; make
 ============================================================================
 */
#include <QThread>
#include <QWaitCondition>
#include <QMutexLocker>
#include <QSemaphore>
#include <QMutex>
#include <iostream>
#include <unistd.h>

using namespace std;

const int BUFFSIZE = 100;
//************************************************************

template<typename T>
class Monitor {
private:
    QMutex l;
    QWaitCondition full, empty;
    int in, out;
    int N;
    int count;
    T *buffer;
public:
    void put(T);
    T get();
    Monitor(int n = BUFFSIZE);
    ~Monitor();
};
//-----------------------------------------

template<typename T>
Monitor<T>::Monitor(int n) {
    buffer = new T[n];
    N = n;
    count = 0;
    in = out = 0;
}
//-----------------------------------------

template<typename T>
Monitor<T>::~Monitor() {
    delete []buffer;
}
//-----------------------------------------

template<typename T>
void Monitor<T>::put(T i) {
    QMutexLocker ml(&l);
    while (count == N)
        full.wait(&l);
    buffer[in] = i;
    in = (in + 1) % N;
    count++;
    empty.wakeOne();
}
//-----------------------------------------

template<typename T>
T Monitor<T>::get() {
    QMutexLocker ml(&l);
    while (count == 0)
        empty.wait(&l);
    T temp = buffer[out];
    out = (out + 1) % N;
    count--;
    full.wakeOne();
    return temp;
}
//************************************************************

template<typename T>
class Producer : public QThread {
private:
    static QSemaphore numProducts;
    int ID;
    static Monitor<T> *mon;
public:
    static T(*produce)();
    static void initClass(int numP, Monitor<T> *m, T(*prod)());

    Producer<T>(int i) : ID(i) {}
    void run();
};
//---------------------------------------
template<typename T> QSemaphore Producer<T>::numProducts;
template<typename T> Monitor<T> * Producer<T>::mon;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T> void Producer<T>::initClass(int numP, Monitor<T> *m, T(*prod)()) {
    mon = m;
    numProducts.release(numP);
    produce = prod;
}
//---------------------------------------  

template<typename T>
void Producer<T>::run() {
    while (numProducts.tryAcquire()) {
        T item = (*produce)();
        mon->put(item);
    }
}
//---------------------------------------  

template<typename T>
class Consumer : public QThread {
private:
    int ID;
    static Monitor<T> *mon;
    static QSemaphore numProducts;
public:
    static void (*consume)(T i);
    static void initClass(int numP, Monitor<T> *m, void (*cons)(T));

    Consumer<T>(int i) : ID(i) {}
    void run();
};
//---------------------------------------

template<typename T> QSemaphore Consumer<T>::numProducts;
template<typename T> Monitor<T> *Consumer<T>::mon;
template<> void (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> void Consumer<T>::initClass(int numP, Monitor<T> *m, void (*cons)(T)) {
    numProducts.release(numP);
    mon = m;
    consume = cons;
}
//---------------------------------------

template<typename T> void Consumer<T>::run() {
    while (numProducts.tryAcquire()) {
        T item = mon->get(); // take the item out
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
    Monitor<int> m;

    Producer<int>::initClass(numP, &m, &produce);
    Consumer<int>::initClass(numP, &m, &consume);

    Producer<int> *p[N];
    Consumer<int> *c[M];
    for (int i = 0; i < N; i++) {
        p[i] = new Producer<int>(i);
        p[i]->start();
    }
    for (int i = 0; i < M; i++) {
        c[i] = new Consumer<int>(i);
        c[i]->start();
    }

    for (int i = 0; i < N; i++)
        p[i]->wait();

    for (int i = 0; i < M; i++)
        c[i]->wait();
    return 0;
}
