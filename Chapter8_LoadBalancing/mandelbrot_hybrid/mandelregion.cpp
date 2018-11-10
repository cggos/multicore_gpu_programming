#include "mandelregion.h"
#include "workqueue.h"
#include "kernel.h"
#include <QColor>
#include <QAtomicInt>

//--------------------------------------
QRgb *MandelRegion::colormap;
double MandelRegion::diffThresh;
int MandelRegion::pixelSizeThresh;
//--------------------------------------
void MandelRegion::initColorMapAndThrer (int maxV, double diffT=0.3, int pixT=2048)
{
  colormap = new QRgb[maxV];
  diffThresh = diffT;
  pixelSizeThresh = pixT;
  
  for (int i = 0; i < maxV; i++)
    {
      double ratio = i * 1.0l / maxV;
      ratio = (i % 256) / 255.0;
      int R, G, B;
      R = 255 * (1 - ratio);
      B = 255 * ratio;
      if (ratio >= 0.5)
        G = 255 - 255 * (ratio - 0.5);
      else
        G = 255 * ratio * 2;
      colormap[i] = qRgb (R, G, B);
    }
}

//--------------------------------------

void MandelRegion::print ()
{
  cout << "Coord(" << imageX << ", " << imageY << ")   Size : " << pixelsX << " x " << pixelsY << endl;
}

//--------------------------------------
int MandelRegion::diverge (double cx, double cy)
{
  int MAXITER = ownerFrame->MAXITER;
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4)
    {
      tx = vx * vx - vy * vy + cx;
      ty = 2 * vx * vy + cy;
      vx = tx;
      vy = ty;
      iter++;
    }
  return iter;
}

//--------------------------------------

MandelRegion::MandelRegion (double uX, double uY, double lX, double lY, int iX, int iY, int pX, int pY, MandelFrame * f)
{
  upperX = uX;
  upperY = uY;
  lowerX = lX;
  lowerY = lY;
  imageX = iX;
  imageY = iY;
  pixelsX = pX;
  pixelsY = pY;
  cornersIter[0] = cornersIter[1] = cornersIter[2] = cornersIter[3] = UNKNOWN;
  ownerFrame = f;
}

//--------------------------------------

void MandelRegion::compute (bool onGPU)
{
  double stepX = ownerFrame->stepX;
  double stepY = ownerFrame->stepY;
  QImage *img = ownerFrame->img;
  int MAXGRAY = ownerFrame->MAXITER;

  if (onGPU)
    {
      unsigned int *h_res;
      int pitch;

      hostFE (upperX, upperY, lowerX, lowerY, pixelsX, pixelsY, &h_res, &pitch, MAXGRAY);
      pitch /= sizeof (int);

      //copy results into QImage object   
      for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++)
          {
            int color = h_res[j * pitch + i];
            if (color == MAXGRAY)
              img->setPixel (imageX + i, imageY + j, qRgb (0, 0, 0));
            else
              img->setPixel (imageX + i, imageY + j, colormap[color]);
          }

    }
  else                          // CPU execution
    {
      for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++)
          {
            double tempx, tempy;
            tempx = upperX + i * stepX;
            tempy = upperY - j * stepY;
            int color = diverge (tempx, tempy);
            if (color == MAXGRAY)
              img->setPixel (imageX + i, imageY + j, qRgb (0, 0, 0));
            else
              img->setPixel (imageX + i, imageY + j, colormap[color]);
          }
    }
}

//--------------------------------------
// if the region is small enough, process it, or split it in 4 regions
void MandelRegion::examine (WorkQueue & q, bool onGPU = false)
{
  int minIter = INT_MAX, maxIter = 0;

  // evaluate the corners first
  for (int i = 0; i < 4; i++)
    {
      if (cornersIter[i] == UNKNOWN)
        {
          switch (i)
            {
            case (UPPER_RIGHT):
              cornersIter[i] = diverge (lowerX, upperY);
              break;
            case (UPPER_LEFT):
              cornersIter[i] = diverge (upperX, upperY);
              break;
            case (LOWER_RIGHT):
              cornersIter[i] = diverge (lowerX, lowerY);
              break;
            default:           // LOWER_LEFT
              cornersIter[i] = diverge (upperX, lowerY);
            }
        }
      if (minIter > cornersIter[i])
        minIter = cornersIter[i];
      else if (maxIter < cornersIter[i])
        maxIter = cornersIter[i];
    }


  // either compute the pixels or break the region in 4 pieces
  if (maxIter - minIter < diffThresh * maxIter || pixelsX * pixelsY < pixelSizeThresh)
    {
      compute (onGPU);
      ownerFrame->regionComplete ();
    }
  else
    {
      double midDiagX1, midDiagY1;      // data for determining the new subregions
      double midDiagX2, midDiagY2;
      int subimageX, subimageY;
      subimageX = pixelsX / 2;  // concern the upper left quad. 
      subimageY = pixelsY / 2;
      midDiagX1 = upperX + (subimageX - 1) * ownerFrame->stepX;
      midDiagY1 = upperY - (subimageY - 1) * ownerFrame->stepY;
      midDiagX2 = midDiagX1 + ownerFrame->stepX;
      midDiagY2 = midDiagY1 - ownerFrame->stepY;

      MandelRegion *sub[4];
      sub[UPPER_LEFT] = new MandelRegion (upperX, upperY, midDiagX1, midDiagY1, imageX, imageY, subimageX, subimageY, ownerFrame);
      sub[UPPER_LEFT]->cornersIter[UPPER_LEFT] = cornersIter[UPPER_LEFT];

      sub[UPPER_RIGHT] = new MandelRegion (midDiagX2, upperY, lowerX, midDiagY1, imageX + subimageX, imageY, pixelsX - subimageX, subimageY, ownerFrame);
      sub[UPPER_RIGHT]->cornersIter[UPPER_RIGHT] = cornersIter[UPPER_RIGHT];

      sub[LOWER_LEFT] = new MandelRegion (upperX, midDiagY2, midDiagX1, lowerY, imageX, imageY + subimageY, subimageX, pixelsY - subimageY, ownerFrame);
      sub[LOWER_LEFT]->cornersIter[LOWER_LEFT] = cornersIter[LOWER_LEFT];

      sub[LOWER_RIGHT] = new MandelRegion (midDiagX2, midDiagY2, lowerX, lowerY, imageX + subimageX, imageY + subimageY, pixelsX - subimageX, pixelsY - subimageY, ownerFrame);
      sub[LOWER_RIGHT]->cornersIter[LOWER_RIGHT] = cornersIter[LOWER_RIGHT];

      for (int i = 0; i < 4; i++)
        {
          q.append (sub[i]);
          //sub[i]->print();
        }
      ownerFrame->regionSplit ();
    }
}

//--------------------------------------
bool MandelRegion::operator< (const MandelRegion & a)
{
  // cout << "Comparing " << this->pixelsX << " " << a.pixelsX<< endl; 
  int Npixels4a = a.pixelsX * a.pixelsY;
  int Npixels4b = this->pixelsX * this->pixelsY;
  return Npixels4a > Npixels4b;
}
