/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Multi-threaded image matching, using non-static methods
 To compile    : Have to replace main.cpp with this file and run : qmake imageMatch.pro; make
 ============================================================================
 */
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <boost/bind.hpp>
#include <QtConcurrentMap>
#include <QThreadStorage>
#include <QVector>
#include <QTime>

using namespace std;

#define MAXLEVELS 2048
//***********************************************************
class Image
{
private:
  int width;
  int height;
  int levels;
  unsigned int *pixel;
  double *p;                    // probabilities
  double MI;                    // mutual information with a target image
  char filename[20];

  // joint probs. This is a per-thread value
  static QThreadStorage < double *>jointProb;
  void calcJointProb (Image * x);

public:
    Image (char *);
   ~Image ();
  static double mutualInformation (Image *, Image *);
  void calcProb();
  double getMI ()
  {
    return MI;
  }
  char *getFilename ()
  {
    return filename;
  }
};

QThreadStorage < double *>Image::jointProb;
//-----------------------------------------------------------
// Used to compare and sort images in descending order of
// the mutual information calculated
bool comp (Image * x, Image * y)
{
  return x->getMI () > y->getMI ();
}

//-----------------------------------------------------------
// Assumes that the file does not contain any comment lines starting with #
// Allocates the memory for the pixel values and the value probabilities
Image::Image (char *fname)
{
  FILE *fin;
  strncpy (filename, fname, 20);
  filename[19] = 0;
  fin = fopen (fname, "rb");
  fscanf (fin, "%*s%i%i%i", &(width), &(height), &(levels));

  pixel = new unsigned int[width * height];
  // first set all values to 0. This is needed as in 2 of the 3 cases
  // only a part of each pixel value is read from the file
  memset ((void *) pixel, 0, width * height * sizeof (unsigned int));
  if (levels < 256)             // each pixel is 1 byte
    for (int i = 0; i < width * height; i++)
      fread ((void *) &(pixel[i]), sizeof (unsigned char), 1, fin);
  else if (levels < 65536)      // each pixel is 2 bytes
    for (int i = 0; i < width * height; i++)
      fread ((void *) &(pixel[i]), sizeof (unsigned short), 1, fin);
  else                          // each pixel is 4 bytes
    fread (pixel, sizeof (unsigned int), width * height, fin);

  levels++;
  fclose (fin);
  p = new double[levels];
}

//-----------------------------------------------------------
// Releases memory
Image::~Image ()
{
  if (pixel != NULL)
    {
      delete[]pixel;
      delete[]p;
      pixel = NULL;
      p = NULL;
    }
}

//-----------------------------------------------------------
void Image::calcProb ()
{
  int numPixels = this->width * this->height;

  // first set all values to 0
  memset ((void *) this->p, 0, this->levels * sizeof (double));
  for (int i = 0; i < numPixels; i++)
    this->p[this->pixel[i]]++;

  for (int i = 0; i < this->levels; i++)
    this->p[i] /= numPixels;
}

//-----------------------------------------------------------
// Precondition : images must have the same spatial resolution and number of grayscale levels
void Image::calcJointProb (Image * x)
{
  double *pij;
  if (jointProb.hasLocalData ())        // joint probabilities storage exist, retrieve its location
    {
      pij = jointProb.localData ();
    }
  else                          // otherwise allocate it and store its address
    {
      pij = new double[MAXLEVELS * MAXLEVELS];
      jointProb.setLocalData (pij);
    }

  int numPixels = width * height;

  // first set all values to 0
  memset ((void *) pij, 0, x->levels * x->levels * sizeof (double));
  for (int i = 0; i < numPixels; i++)
    pij[pixel[i] * x->levels + x->pixel[i]]++;

  for (int i = 0; i < x->levels * x->levels; i++)
    pij[i] /= numPixels;
}

//-----------------------------------------------------------
// The probabilities must be calculated before hand
double Image::mutualInformation (Image * x, Image * y)
{
  x->calcJointProb (y);
  double *pij = jointProb.localData (); // the array has been created already by the previous statement
  double mutual = 0;
  for (int i = 0; i < x->levels; i++)
    for (int j = 0; j < y->levels; j++)
      {
        int idx = i * y->levels + j;
        if (x->p[i] != 0 && y->p[j] != 0 && pij[idx] != 0)
          mutual += pij[idx] * log (pij[idx] / (x->p[i] * y->p[j]));
      }
  x->MI = mutual / log (2);
  return x->MI;
}


struct Wrapper
{
  void operator()(Image *x)
  {
      x->calcProb();
  }
};
//***********************************************************
int main (int argc, char *argv[])
{
  int numImages = atoi (argv[1]);
  QTime t;
  t.start ();

  // read the target and all other images
  Image target ("images/main.pgm");     // target image
  QVector < Image * >pool;
  for (int picNum = 0; picNum < numImages; picNum++)
    {
      char buff[100];
      sprintf (buff, "images/(%i).pgm", picNum);
      pool.append (new Image (buff));
    }
  int iodone = t.elapsed ();

  // pixel value probabilities calculation
  target.calcProb ();

  Wrapper w;
  QtConcurrent::blockingMap (pool, w);

  // mutual information (MI) calculation
  QtConcurrent::blockingMap (pool, boost::bind (Image::mutualInformation, _1, &target));

  // sorting of the images according to MI findings
  qSort (pool.begin (), pool.end (), comp);
  printf ("%i %i\n", iodone, t.elapsed () - iodone);

  return 0;
}
