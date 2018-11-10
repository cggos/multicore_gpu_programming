/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To compile    : qmake mapped_result_access.pro; make
 ============================================================================
 */
#include <QtConcurrent/QtConcurrentMap>
#include <QList>
#include <vector>
#include <iostream>

using namespace std;

int mult(const int x)
{
   return x*2;
}

int main(int argc, char *argv[])
{
    vector<int> data;
    for(int i=0;i<100;i++)
        data.push_back(i);

    QFuture<int> r = QtConcurrent::mapped(data, mult);
    QList<int> res = r.results();
    for(int i=0;i<res.size(); i++)
        cout << res[i] << " ";
    cout << endl;

    for(QFuture<int>::const_iterator i = r.begin(); i != r.end(); i++)
        cout << *i << " ";
    cout << endl;

    for(int i=0;i<r.resultCount(); i++)
        cout << r.resultAt(i) << " ";
    cout << endl;

    return 0;
}
