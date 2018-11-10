// Program is modified for timing purposes
// G. Barlas, 5/2014

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <QTime>
#include <assert.h>
#include "color_conversion.cpp"

using namespace std;

int X, Y;
float *L, *a, *b, *new_L;
int *R, *G, *B;

//********************************************************************
// helper function for boundary pixels
float pixel_L (int x, int y)
{
  if (x >= 0 && x < X && y >= 0 && y < Y)
    return L[y * X + x];
  else
    return 0;
}

//********************************************************************
int main (int argc, char **argv)
{
  fstream fin, fout;
  char buff[80];
  bool binflag;
  int max_value, i, j, loc, maxiter = 1;
  int maxPixels;

  if (argc < 3)
    {
      cerr << "Usage : " << argv[0] << " inFile outFile [iterations] [pixels]\n";
      exit (1);
    }

  //------------------------------------------
  // Get execution parameters
  if (argc > 3)
    maxiter = atoi (argv[3]);

  fin.open (argv[1], ios::in | ios::binary);
  assert (fin);
  fin.getline (buff, 80);
  binflag = !strcmp (buff, "P6");

  fin.getline (buff, 80);
  if (buff[0] == '#')
    fin.getline (buff, 80);
  X = atoi (buff);
  strtok (buff, " ");
  Y = atoi (strtok (NULL, " "));


  if (argc > 4)
    maxPixels = atoi (argv[4]);
  else
    maxPixels = X * Y;

  R = new int[X * Y];
  G = new int[X * Y];
  B = new int[X * Y];
  memset ((void *) R, 0, X * Y * sizeof (int));
  memset ((void *) G, 0, X * Y * sizeof (int));
  memset ((void *) B, 0, X * Y * sizeof (int));

  L = new float[X * Y];
  a = new float[X * Y];
  b = new float[X * Y];
  new_L = new float[X * Y];
  assert (R != NULL && G != NULL && B != NULL && L != NULL && a != NULL && b != NULL && new_L != NULL);
  fin.getline (buff, 80);
  max_value = atoi (buff);

  if (binflag)
    {
      for (j = 0; j < Y; j++)
        for (i = 0; i < X; i++)
          {
            loc = j * X + i;
            fin.read ((char *) &R[loc], 1);
            fin.read ((char *) &G[loc], 1);
            fin.read ((char *) &B[loc], 1);
          }
    }
  else
    {
      for (j = 0; j < Y; j++)
        for (i = 0; i < X; i++)
          {
            loc = j * X + i;
            fin >> R[loc] >> G[loc] >> B[loc];
          }
    }

  fin.close ();


  // timing variables
  QTime t;
  t.start();
  int iter = maxiter;
  while (iter--)
    {
      //------------------------------------------
      // main body : RGB -> LAB conversion
      int pixelLimit = 0;
      for (j = 0; j < Y; j++)
        for (i = 0; i < X; i++)
          {
            loc = j * X + i;
            RGB2LAB (R[loc], G[loc], B[loc], L[loc], a[loc], b[loc]);

            pixelLimit++;
            if (pixelLimit > maxPixels)
              break;
          }

      // convolution
      pixelLimit = 0;
      for (j = 0; j < Y; j++)
        for (i = 0; i < X; i++)
          {
            double temp;
            temp = 5.0 * pixel_L (i, j);
            temp -= (pixel_L (i - 1, j) + pixel_L (i + 1, j) + pixel_L (i, j - 1) + pixel_L (i, j + 1));

            loc = j * X + i;
            new_L[loc] = round (temp);
            if (new_L[loc] <= 0)
              new_L[loc] = 0;

            pixelLimit++;
            if (pixelLimit > maxPixels)
              break;
          }


      // LAB -> RGB inverse color conversion
      pixelLimit = 0;
      for (j = 0; j < Y; j++)
        for (i = 0; i < X; i++)
          {
            loc = j * X + i;
            LAB2RGB (new_L[loc], a[loc], b[loc], R[loc], G[loc], B[loc]);

            // fix conversion under-/over-flow
            // Just a shortcut replacing proper normalization
            if (R[loc] < 0)
              R[loc] = 0;
            else if (R[loc] > max_value)
              R[loc] = max_value;
            if (G[loc] < 0)
              G[loc] = 0;
            else if (G[loc] > max_value)
              G[loc] = max_value;
            if (B[loc] < 0)
              B[loc] = 0;
            else if (B[loc] > max_value)
              B[loc] = max_value;

            pixelLimit++;
            if (pixelLimit > maxPixels)
              break;

          }
    }
  cout << "Elapsed time per convolution for " << maxPixels << " pixels " << t.elapsed()*0.001/maxiter << endl;
  
  //-------------------------------------------    
  // save sharpened file
  fout.open (argv[2], ios::out | ios::binary);
  assert (fout);
  fout << "P6\n" << "#Sharpened image\n" << X << " " << Y << endl;
  fout << max_value << endl;

  for (j = 0; j < Y; j++)
    for (i = 0; i < X; i++)
      {
        loc = j * X + i;
        fout.write ((char *) &R[loc], 1);
        fout.write ((char *) &G[loc], 1);
        fout.write ((char *) &B[loc], 1);
      }

  fout.close ();

  // release memory
  delete[]new_L;
  delete[]b;
  delete[]a;
  delete[]L;
  delete[]B;
  delete[]G;
  delete[]R;

  return 0;
}
