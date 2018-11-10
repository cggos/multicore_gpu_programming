/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : September 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake monitor2ProdCons.pro; make
 ============================================================================
 */
#include <QThread>
#include <QWaitCondition>
#include <QMutexLocker>
#include <QSemaphore>
#include <QMutex>
#include <iostream>
#include <unistd.h>
#include <queue>

using namespace std;

const int BUFFSIZE = 100;
//************************************************************

template<typename T>
class Monitor {
private:
    QMutex l;
    QWaitCondition full, empty;
    queue<T *> emptySpotsQ;
    queue<T *> itemQ;
    T *buffer;
public:
    T* canPut();
    T* canGet();
    void donePutting(T *x);
    void doneGetting(T *x);
    Monitor(int n = BUFFSIZE);
    ~Monitor();
};
//-----------------------------------------

template<typename T>
Monitor<T>::Monitor(int n) {
    buffer = new T[n];
    for(int i=0;i<n;i++)
        emptySpotsQ.push(&buffer[i]);
}
//-----------------------------------------

template<typename T>
Monitor<T>::~Monitor() {
    delete []buffer;
}
//-----------------------------------------

template<typename T>
T* Monitor<T>::canPut() {
    QMutexLocker ml(&l);
    while (emptySpotsQ.size() == 0)
        full.wait(&l);
    T *aux = emptySpotsQ.front();
    emptySpotsQ.pop();
    return aux;
}
//-----------------------------------------

template<typename T>
T* Monitor<T>::canGet() {
    QMutexLocker ml(&l);
    while (itemQ.size() == 0)
        empty.wait(&l);
    T* temp = itemQ.front();
    itemQ.pop();
    return temp;
}
//-----------------------------------------

template<typename T>
void Monitor<T>::donePutting(T *x) {
    QMutexLocker ml(&l);
    itemQ.push(x);
    empty.wakeOne();
}
//-----------------------------------------

template<typename T>
void Monitor<T>::doneGetting(T *x) {
    QMutexLocker ml(&l);
    emptySpotsQ.push(x);
    full.wakeOne();
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
        T* aux = mon->canPut();
        *aux = item;
        mon->donePutting(aux);
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
        T* aux = mon->canGet();
        T item = *aux; // take the item out
        mon->doneGetting(aux);
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
    cout << i;
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
