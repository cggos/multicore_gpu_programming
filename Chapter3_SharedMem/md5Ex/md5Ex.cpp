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
#include <cstdio>
#include <string>
#include <unistd.h>
#include "md5.h"
#include <QtConcurrentRun>
#include <QFuture>

using namespace std;

int main (int argc, char *argv[])
{
  int N = argc - 1;
  QFuture < string > f[N];
  char *buffers[N];

  // scan all the filenames supplied
  for (int i = 0; i < N; i++)
    {
      buffers[i] = 0;
      FILE *fin;
      fin = fopen (argv[i + 1], "r");
      if (fin != 0)      // if the file exists
        {
          fseek (fin, 0, SEEK_END);
          int fileSize = ftell (fin);   // find out the size
          fseek (fin, 0, SEEK_SET);

          buffers[i] = new char[fileSize + 1];  // allocate enough memory
          fread (buffers[i], fileSize, 1, fin); // read all of it in memory
          buffers[i][fileSize + 1] = 0;         // terminate by 0 as md5() expects a string
          fclose (fin);
          string s (buffers[i]);
          f[i] = QtConcurrent::run (md5, s);    // calculate the MD5 hash in another thread
        }
    }


  for (int i = 0; i < N; i++)
    if (buffers[i] != 0)         // if file existed
      {
        f[i].waitForFinished (); // wait for the calculation to complete
        cout << argv[i + 1] << " : " << f[i].result () << endl;
        delete[]buffers[i];      // cleanup the buffer
      }
  return 0;
}
