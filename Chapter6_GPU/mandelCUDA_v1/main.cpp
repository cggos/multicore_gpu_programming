/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : make
 ============================================================================
 */
#include <QImage>
#include <QRgb>
#include <QTime>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "kernel.h"

using namespace std;

//************************************************************

int main (int argc, char *argv[])
{
  double upperCornerX, upperCornerY;
  double lowerCornerX, lowerCornerY;

  upperCornerX = atof (argv[1]);
  upperCornerY = atof (argv[2]);
  lowerCornerX = atof (argv[3]);
  lowerCornerY = atof (argv[4]);

  // support for timing the operation
  int iterations = 1;
  if (argc > 5)
    iterations = atoi(argv[5]);

  int imgX = 1024, imgY = 768;
  QImage *img = new QImage (imgX, imgY, QImage::Format_RGB32);

  QTime t;
  t.start();
  
  int i = iterations;
  while (i--)
    hostFE (upperCornerX, upperCornerY, lowerCornerX, lowerCornerY, img, imgX, imgY);

  cout << "Time (ms) per iteration " << t.elapsed()*1.0/iterations << endl;
  
  img->save ("mandel.png", "PNG", 0);
  return 0;
}
