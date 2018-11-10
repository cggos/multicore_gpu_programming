#include "mandelframe.h"

//----------------------------------------------------------------------
MandelFrame::MandelFrame(double uX, double uY, double lX, double lY, int pX, int pY, char *c, int maxiter) {
    upperX = uX;
    upperY = uY;
    lowerX = lX;
    lowerY = lY;
    img = new QImage(pX, pY, QImage::Format_RGB32);
    memset(fname,0,MAXFNAME);
    strncpy(fname, c, MAXFNAME);
    MAXITER = maxiter;
    pixelsX = pX;
    pixelsY = pY;
    stepX = (lowerX - upperX) / pixelsX;
    stepY = (upperY - lowerY) / pixelsY;
    remainingRegions = 1;  // when this becomes 0, the frame has been calculated
}
//----------------------------------------------------------------------
void MandelFrame::regionSplit()
{
  remainingRegions.fetchAndAddOrdered(3);
}
//----------------------------------------------------------------------
void MandelFrame::regionComplete()
{
  if(remainingRegions.fetchAndAddOrdered(-1) == 1) // if last was 1 now it is 0
  {
     img->save(fname,"PNG");
  }
}
