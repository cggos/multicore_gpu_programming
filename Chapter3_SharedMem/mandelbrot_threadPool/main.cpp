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
#include <QImage>
#include <QRgb>
#include "customThreadPool.h"
#include <stdlib.h>
#include <iostream>

using namespace std;


//************************************************************

class MandelCompute : public ComputationalTask {
private:
    static const int MAXITER;
    int diverge(double cx, double cy);

    double upperX, upperY, lowerX, lowerY;
    int imageX, imageY, pixelsX, pixelsY;
    QImage *img;

public:
    MandelCompute(double, double, double, double, QImage *, int, int, int, int);
    virtual void compute();
};
const int MandelCompute::MAXITER = 255;

//--------------------------------------

int MandelCompute::diverge(double cx, double cy) {
    int iter = 0;
    double vx = cx, vy = cy, tx, ty;
    while (iter < MAXITER && (vx * vx + vy * vy) < 4) {
        tx = vx * vx - vy * vy + cx;
        ty = 2 * vx * vy + cy;
        vx = tx;
        vy = ty;
        iter++;
    }
    return iter;
}
//--------------------------------------

MandelCompute::MandelCompute(double uX, double uY, double lX, double lY, QImage *im, int iX, int iY, int pX, int pY) {
    upperX = uX;
    upperY = uY;
    lowerX = lX;
    lowerY = lY;
    img = im;
    imageX = iX;
    imageY = iY;
    pixelsX = pX;
    pixelsY = pY;
}

//--------------------------------------

void MandelCompute::compute() {
    double stepx = (lowerX - upperX) / pixelsX;
    double stepy = (upperY - lowerY) / pixelsY;

    for (int i = 0; i < pixelsX; i++)
        for (int j = 0; j < pixelsY; j++) {
            double tempx, tempy;
            tempx = upperX + i * stepx;
            tempy = upperY - j * stepy;
            int color = diverge(tempx, tempy);
            img->setPixel(imageX + i, imageY + j, qRgb(256 - color, 256 - color, 256 - color));
        }
}
//************************************************************

int main(int argc, char *argv[]) {
    double upperCornerX, upperCornerY;
    double lowerCornerX, lowerCornerY;

    upperCornerX = atof(argv[1]);
    upperCornerY = atof(argv[2]);
    lowerCornerX = atof(argv[3]);
    lowerCornerY = atof(argv[4]);
    double partXSpan, partYSpan;


    int Xparts = 10, Yparts = 10;
    int imgX = 1024, imgY = 768;
    int pxlX, pxlY;

    partXSpan = (lowerCornerX - upperCornerX) / Xparts;
    partYSpan = (upperCornerY - lowerCornerY) / Yparts;
    pxlX = imgX / Xparts;
    pxlY = imgY / Yparts;
    QImage *img = new QImage(imgX, imgY, QImage::Format_RGB32);
    CustomThreadPool tp;
    unsigned thIDs[Xparts][Yparts];



    for (int i = 0; i < Xparts; i++)
        for (int j = 0; j < Yparts; j++) {
            double x1, y1, x2, y2;
            int ix, iy, pX, pY; //image coords. and pixel spans

            x1 = upperCornerX + i * partXSpan;
            y1 = upperCornerY - j * partYSpan;
            x2 = upperCornerX + (i + 1) * partXSpan;
            y2 = upperCornerY - (j + 1) * partYSpan;

            ix = i*pxlX;
            iy = j*pxlY;
            pX = (i == Xparts - 1) ? imgX - ix : pxlX;
            pY = (j == Yparts - 1) ? imgY - iy : pxlY;
            
            MandelCompute *t = new MandelCompute(x1, y1, x2, y2, img, ix, iy, pX, pY);
            thIDs[i][j] = tp.schedule(t);
        }


    // now wait for all threads to stop
    for (int i = 0; i < Xparts; i++)
        for (int j = 0; j < Yparts; j++) {
            tp.waitTillDone(thIDs[i][j]);
        }

    img->save("mandel.png", "PNG", 0);
    return 0;
}
