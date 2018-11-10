/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Sequential image matching
 To compile    : qmake seq.pro; make
 ============================================================================
 */
// Sequential version
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
  double MI;
  char filename[20];
  //static double pij[MAXLEVELS * MAXLEVELS]; // joined probs.
  static QThreadStorage < double *>jointProb;
  void calcJointProb (Image * x);

public:
    Image (char *);
   ~Image ();
//   double mutualInformation(Image *);
  static double mutualInformation (Image *, Image *);
  static void calcProb (Image *);
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
bool comp (Image * x, Image * y)
{
  return x->getMI () > y->getMI ();
}

//-----------------------------------------------------------
Image::Image (char *fname)
{
  FILE *fin;
  strncpy (filename, fname, 20);
  filename[19] = 0;
  fin = fopen (fname, "rb");
  fscanf (fin, "%*s%i%i%i", &(width), &(height), &(levels));

  pixel = new unsigned int[width * height];
  memset ((void *) pixel, 0, width * height * sizeof (unsigned int));
  if (levels < 256)
    for (int i = 0; i < width * height; i++)
      fread ((void *) &(pixel[i]), sizeof (unsigned char), 1, fin);
  else if (levels < 65536)
    for (int i = 0; i < width * height; i++)
      fread ((void *) &(pixel[i]), sizeof (unsigned short), 1, fin);
  else
    fread (pixel, sizeof (unsigned int), width * height, fin);

  levels++;
  fclose (fin);
  p = new double[levels];
}

//-----------------------------------------------------------
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
void Image::calcProb (Image * x)
{
  int numPixels = x->width * x->height;

  memset ((void *) x->p, 0, x->levels * sizeof (double));
  for (int i = 0; i < numPixels; i++)
    x->p[x->pixel[i]]++;

  for (int i = 0; i < x->levels; i++)
    x->p[i] /= numPixels;
}

//-----------------------------------------------------------
// Precondition : images must have the same spatial resolution and number of grayscale levels
void Image::calcJointProb (Image * x)
{
  double *pij;
  if (jointProb.hasLocalData ())
    {
      pij = jointProb.localData ();
    }
  else
    {
      pij = new double[MAXLEVELS * MAXLEVELS];
      jointProb.setLocalData (pij);
    }

  int numPixels = width * height;
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
  double *pij = jointProb.localData (); // the array has been created already
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

//***********************************************************
int main (int argc, char *argv[])
{
  int numImages = atoi (argv[1]);
  QTime t;
  t.start ();

  Image target ("images/main.pgm");     // target image
  QVector < Image * >pool;
  for (int picNum = 0; picNum < numImages; picNum++)
    {
      char buff[100];
      sprintf (buff, "images/(%i).pgm", picNum);
      pool.append (new Image (buff));
    }

  int iodone = t.elapsed ();

  Image::calcProb (&target);
  for (int picNum = 0; picNum < numImages; picNum++)
    {
      Image::calcProb (pool[picNum]);
      Image::mutualInformation (pool[picNum], &target);
    }

  qSort (pool.begin (), pool.end (), comp);
  printf ("%i %i\n", iodone, t.elapsed () - iodone);

  return 0;
}
