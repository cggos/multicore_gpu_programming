#ifndef MANDELREGION_H
#define MANDELREGION_H

#include <QImage>
#include <QRgb>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include "mandelframe.h"

using namespace std;

const int UNKNOWN = -1;
const int UPPPER_RIGHT = 0;
const int UPPPER_LEFT = 1;
const int LOWER_RIGHT = 2;
const int LOWER_LEFT = 3;

class WorkQueue;
//************************************************************

class MandelRegion
{
private:
  int diverge (double cx, double cy);

  double upperX, upperY, lowerX, lowerY;
  int imageX, imageY, pixelsX, pixelsY;
  int cornersIter[4];
  MandelFrame *ownerFrame;
  static QRgb *colormap;
  static double diffThresh;
  static int pixelSizeThresh;
  
public:
    MandelRegion (double, double, double, double, int, int, int, int, MandelFrame *);
  void compute (bool onGPU);
  void examine (WorkQueue &, bool onGPU);
  void print ();
  bool operator< (const MandelRegion & a);
  static void initColorMapAndThrer (int, double, int);

  struct Compare
  {
    bool operator () (const MandelRegion * a, const MandelRegion * b)
    {
      int Npixels4a = a->pixelsX * a->pixelsY;
      int Npixels4b = b->pixelsX * b->pixelsY;
        return Npixels4a > Npixels4b;
    }

  };
};
#endif
