/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake terminProdCons2.pro; make
 ============================================================================
 */
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>

using namespace std;

const int BUFFSIZE = 100;

template<typename T>
class Producer : public QThread {
private:
    int ID;
    static QSemaphore *slotsAvail;
    static QSemaphore *resAvail;
    static QMutex l1;
    static T *buffer;
    static volatile bool *exitFlag;
    static int in;
public:
    static T(*produce)();
    static void initClass(QSemaphore *s, QSemaphore *a, T *b, T(*prod)(), bool *e);

    Producer<T>(int i) : ID(i) {
    }
    void run();
};
//---------------------------------------
template<typename T> QSemaphore * Producer<T>::slotsAvail;
template<typename T> QSemaphore * Producer<T>::resAvail;
template<typename T> QMutex Producer<T>::l1;
template<typename T> T * Producer<T>::buffer;
template<typename T> volatile bool *Producer<T>::exitFlag;
template<typename T> int Producer<T>::in = 0;
template<> int (*Producer<int>::produce)() = NULL;
//---------------------------------------

template<typename T>
void Producer<T>::initClass(QSemaphore *s, QSemaphore *a, T *b, T(*prod)(), bool *e) {
    slotsAvail = s;
    resAvail = a;
    buffer = b;
    produce = prod;
    exitFlag = e;
}
//---------------------------------------

template<typename T>
void Producer<T>::run() {
    while (*exitFlag == false) {
        T item = (*produce)();
        slotsAvail->acquire(); // wait for an empty slot in the buffer

        if (*exitFlag) return; // stop immediately on termination    

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
class Consumer : public QThread {
private:
    int ID;
    static QSemaphore *slotsAvail;
    static QSemaphore *resAvail;
    static QMutex l2;
    static T *buffer;
    static int numConsumers, numProducers;
    static int out;
    static volatile bool *exitFlag;
public:
    static bool (*consume)(T i);
    static void initClass(QSemaphore *s, QSemaphore *a, T* b, bool (*cons)(T), int N, int M, bool *e);
    Consumer<T>(int i) : ID(i) {}
    void run();
};
//---------------------------------------

template<typename T> QSemaphore * Consumer<T>::slotsAvail;
template<typename T> QSemaphore * Consumer<T>::resAvail;
template<typename T> QMutex Consumer<T>::l2;
template<typename T> volatile bool *Consumer<T>::exitFlag;
template<typename T> T * Consumer<T>::buffer;
template<typename T> int Consumer<T>::out = 0;
template<typename T> int Consumer<T>::numConsumers;
template<typename T> int Consumer<T>::numProducers;
template<> bool (*Consumer<int>::consume)(int) = NULL;

//---------------------------------------

template<typename T> 
void Consumer<T>::initClass(QSemaphore *s, QSemaphore *a, T* b, bool (*cons)(T), int N, int M, bool *e) {
    slotsAvail = s;
    resAvail = a;
    consume = cons;
    buffer = b;
    numProducers = N;
    numConsumers = M;
    exitFlag = e;
}
//---------------------------------------

template<typename T> void Consumer<T>::run() {
    while (*exitFlag == false) {
        resAvail->acquire(); // wait for an available item

        if (*exitFlag) return; // stop immediately on termination

        l2.lock();
        int tmpOut = out;
        out = (out + 1) % BUFFSIZE; // update the out index
        l2.unlock();
        T item = buffer[tmpOut]; // take the item out
        slotsAvail->release(); // signal for a new empty slot 

        if ((*consume)(item)) break; // time to stop?
    }

    // only the thread initially detecting termination reaches here
    *exitFlag = true;
    resAvail->release(numConsumers - 1);
    slotsAvail->release(numProducers);
}
//---------------------------------------

int produce() {
    // to be implemented
    int aux = rand();
    return aux;
}
//---------------------------------------

bool consume(int i) {
    // to be implemented
    cout << "@"; // just to show something is happening
    if (i % 10 == 0) return true;
    else return false;
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #producers #consumers\n";
        exit(1);
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int *buffer = new int[BUFFSIZE];
    QSemaphore avail, buffSlots(BUFFSIZE);
    bool exitFlag = false;

    Producer<int>::initClass(&buffSlots, &avail, buffer, &produce, &exitFlag);
    Consumer<int>::initClass(&buffSlots, &avail, buffer, &consume, N, M, &exitFlag);

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

    delete [] buffer;
    return 0;
}
