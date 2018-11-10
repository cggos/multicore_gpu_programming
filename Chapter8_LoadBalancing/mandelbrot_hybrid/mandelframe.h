#ifndef MANDELFRAME_H
#define MANDELFRAME_H

#include <QImage>
#include <QAtomicInt>

const int MAXFNAME=50;

// Used to represent an image frame
class MandelFrame
{
public:
    int MAXITER;
    double upperX, upperY, lowerX, lowerY;
    double stepX, stepY;
    int pixelsX, pixelsY;
    QImage *img;
    char fname[MAXFNAME+1];
    QAtomicInt remainingRegions;
    
    MandelFrame(double, double, double, double, int, int, char *, int );
    void regionSplit();
    void regionComplete();
};

#endif