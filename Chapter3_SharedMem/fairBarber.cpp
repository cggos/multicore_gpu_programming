/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake fairBarber.pro; make
 ============================================================================
 */
#include <QThread>
#include <QSemaphore>
#include <QMutex>
#include <iostream>
#include <unistd.h>

using namespace std;

const int NUMCHAIRS=10;
//---------------------------------------

void concurPrint(int cID, int bID) {
    static QMutex l;
    l.lock();
    cout << "Customer " << cID << " is being serviced by barber " << bID << endl;
    l.unlock();
}
//---------------------------------------

class Barber : public QThread {
private:
    int ID;
    static QSemaphore *barberReady;
    static QSemaphore *customerReady;
    static QSemaphore *barberDone;

    static QMutex l1;
    static QSemaphore customersLeft;
    static int *buffer;
    static int in;
    static int numBarbers;
public:
    static void initClass(int numB, int numC, QSemaphore *r, QSemaphore *c, QSemaphore *d, int *b);

    Barber(int i) : ID(i) {}
    void run();
};
//---------------------------------------
QSemaphore *Barber::barberReady;
QSemaphore *Barber::customerReady;
QSemaphore *Barber::barberDone;
QMutex Barber::l1;
QSemaphore Barber::customersLeft;
int *Barber::buffer;
int Barber::in = 0;
int Barber::numBarbers;
//---------------------------------------

void Barber::initClass(int numB, int numC, QSemaphore *r, QSemaphore *c, QSemaphore *d, int *b) {
    customersLeft.release(numC);
    barberReady = r;
    customerReady = c;
    barberDone = d;
    buffer = b;
    numBarbers = numB;
}
//---------------------------------------  

void Barber::run() {
    while (customersLeft.tryAcquire()) {
        l1.lock();
        buffer[in] = ID;
        in = (in + 1) % numBarbers;
        l1.unlock();
        barberReady->release(); // signal availability
        customerReady[ID].acquire(); // wait for customer to be sitted
        barberDone[ID].release(); // signal that hair is done
    }
}
//---------------------------------------  

class Customer : public QThread {
private:
    int ID;
    static QSemaphore *barberReady;
    static QSemaphore *customerReady;
    static QSemaphore *barberDone;
    static QSemaphore waitChair;
    static QSemaphore barberChair;
    static QMutex l2;
    static int *buffer;
    static int out;
    static int numBarbers;
    static QSemaphore numProducts;
public:
    static void initClass(int numB, QSemaphore *r, QSemaphore *c, QSemaphore *d, int *b);

    Customer(int i) : ID(i) {}
    void run();
};
//---------------------------------------
QSemaphore *Customer::barberReady;
QSemaphore *Customer::customerReady;
QSemaphore *Customer::barberDone;
QSemaphore Customer::waitChair(NUMCHAIRS);
QSemaphore Customer::barberChair;
QMutex Customer::l2;
int * Customer::buffer;
int Customer::out = 0;
int Customer::numBarbers;

//---------------------------------------

void Customer::initClass(int numB, QSemaphore *r, QSemaphore *c, QSemaphore *d, int *b) {
    barberReady = r;
    customerReady = c;
    barberDone = d;
    buffer = b;
    numBarbers = numB;
    barberChair.release(numB);
}
//---------------------------------------

void Customer::run() {
    waitChair.acquire(); // wait for a chair
    barberReady->acquire(); // wait for a barber to be ready
    l2.lock();
    int bID = buffer[out];
    out = (out + 1) % numBarbers;
    l2.unlock();
    waitChair.release(); // get up from the chair
    barberChair.acquire(); // wait for an available barber chair
    customerReady[bID].release(); // signal that customer is ready
    concurPrint(ID, bID);
    barberDone[bID].acquire(); // wait for barber to finish haircut
    barberChair.release(); // get up from barber's chair
}
//---------------------------------------

int main(int argc, char *argv[]) {
    if (argc == 1) {
        cerr << "Usage " << argv[0] << " #barbers #customers\n";
        exit(1);
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int *buffer = new int[N];

    QSemaphore barberReady;
    QSemaphore *customerReady = new QSemaphore[N];
    QSemaphore *barberDone = new QSemaphore[N];

    Barber::initClass(N, M, &barberReady, customerReady, barberDone, buffer);
    Customer::initClass(N, &barberReady, customerReady, barberDone, buffer);

    Barber * p[N];
    Customer * c[M];
    for (int i = 0; i < N; i++) {
        p[i] = new Barber(i);
        p[i]->start();
    }
    for (int i = 0; i < M; i++) {
        c[i] = new Customer(i);
        c[i]->start();
    }

    for (int i = 0; i < N; i++)
        p[i]->wait();

    for (int i = 0; i < M; i++)
        c[i]->wait();

    delete[] buffer;
    delete[] customerReady;
    delete[] barberDone;
    return 0;
}
