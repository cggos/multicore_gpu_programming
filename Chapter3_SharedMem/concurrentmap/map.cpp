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
#include <QtConcurrentMap>
#include <QFuture>
#include <vector>
#include <iostream>

using namespace std;

const double a = 2.0;
const double b = -1.0;
void func(double &x)
{
    x = a*x + b;
}
//-----------------------------------------------
int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    vector<double> data;
    // populate the input data
    for(int i=0;i<N;i++)
        data.push_back(i);
    
    QFuture<void> res = QtConcurrent::map(data.begin(), 
                                          data.end(), 
                                          func);

    res.waitForFinished();
    cout << "Done\n";
    return 0;
}
