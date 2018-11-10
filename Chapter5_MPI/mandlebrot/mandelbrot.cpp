/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : qmake; make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "mpi.h"
#include <QImage>
#include <QRgb>
using namespace std;

//************************************************************
// Communication tags
#define NULLRESULTTAG 0
#define RESULTTAG     1
#define WORKITEMTAG   2
#define ENDTAG        3

//************************************************************

typedef struct WorkItem
{
  double upperX, upperY, lowerX, lowerY;
  int pixelsX, pixelsY, imageX, imageY;
} WorkItem;

//************************************************************
// Class for computing a fractal set part
class MandelCompute
{
private:
  double upperX, upperY, lowerX, lowerY;
  int pixelsX, pixelsY;
  int *img;

  static int MAXITER;
  int diverge (double cx, double cy);

public:
    MandelCompute ();
  void init (WorkItem * wi);
   ~MandelCompute ();
  int *compute ();
};
int MandelCompute::MAXITER = 255;

//--------------------------------------

int MandelCompute::diverge (double cx, double cy)
{
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
MandelCompute::~MandelCompute ()
{
  if (img != NULL)
    delete[]img;
}

//--------------------------------------
MandelCompute::MandelCompute ()
{
  img = NULL;
}

//--------------------------------------

void MandelCompute::init (WorkItem * wi)
{
  upperX = wi->upperX;
  upperY = wi->upperY;
  lowerX = wi->lowerX;
  lowerY = wi->lowerY;

  if (img == NULL || pixelsX != wi->pixelsX || pixelsY != wi->pixelsY)
    {
      if (img != NULL)
        delete[]img;
      img = new int[(wi->pixelsX) * (wi->pixelsY)];
    }
  pixelsX = wi->pixelsX;
  pixelsY = wi->pixelsY;
}

//--------------------------------------

int *MandelCompute::compute ()
{
  double stepx = (lowerX - upperX) / pixelsX;
  double stepy = (upperY - lowerY) / pixelsY;

  for (int i = 0; i < pixelsX; i++)
    for (int j = 0; j < pixelsY; j++)
      {
        double tempx, tempy;
        tempx = upperX + i * stepx;
        tempy = upperY - j * stepy;
        img[j * pixelsX + i] = diverge (tempx, tempy);
      }
  return img;
}

//************************************************************
void registerWorkItem (MPI_Datatype * workItemType)
{
  struct WorkItem sample;

  int blklen[2];
  MPI_Aint displ[2], off, base;
  MPI_Datatype types[2];

  blklen[0] = 4;
  blklen[1] = 2;  // the part's location in the final image is not communicated

  types[0] = MPI_DOUBLE;
  types[1] = MPI_INT;

  displ[0] = 0;
  MPI_Get_address (&(sample.upperX), &base);
  MPI_Get_address (&(sample.pixelsX), &off);
  displ[1] = off - base;

  MPI_Type_create_struct (2, blklen, displ, types, workItemType);
  MPI_Type_commit (workItemType);
}

//************************************************************
// Uses the divergence iterations to pseudocolor the fractal set
void savePixels (QImage * img, int *imgPart, int imageX, int imageY, int height, int width)
{
  for (int i = 0; i < width; i++)
    for (int j = 0; j < height; j++)
      {
        int color = imgPart[j * width + i];
        img->setPixel (imageX + i, imageY + j, qRgb (256 - color, 256 - color, 256 - color));
      }
}

//************************************************************
int main (int argc, char *argv[])
{
  int N, rank;
  double start_time, end_time;
  MPI_Status status;
  MPI_Request request;

  start_time = MPI_Wtime ();

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &N);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  MPI_Datatype workItemType;
  registerWorkItem (&workItemType);


  if (rank == 0)   // master code
    {
      if (argc < 6)
        {
          cerr << argv[0] << " upperCornerX upperCornerY lowerCornerX lowerCornerY workItemPixelsPerSide\n";
          MPI_Abort (MPI_COMM_WORLD, 1);
        }

      double upperCornerX, upperCornerY;
      double lowerCornerX, lowerCornerY;
      double partXSpan, partYSpan;
      int workItemPixelsPerSide;
      int Xparts, Yparts;
      int imgX = 1024, imgY = 768;

      upperCornerX = atof (argv[1]);
      upperCornerY = atof (argv[2]);
      lowerCornerX = atof (argv[3]);
      lowerCornerY = atof (argv[4]);
      workItemPixelsPerSide = atoi (argv[5]);

      // make sure that the image size is evenly divided in work items
      Xparts = (int) ceil (imgX * 1.0 / workItemPixelsPerSide);
      Yparts = (int) ceil (imgY * 1.0 / workItemPixelsPerSide);
      imgX = Xparts * workItemPixelsPerSide;
      imgY = Yparts * workItemPixelsPerSide;

      partXSpan = (lowerCornerX - upperCornerX) / Xparts;
      partYSpan = (upperCornerY - lowerCornerY) / Yparts;
      QImage *img = new QImage (imgX, imgY, QImage::Format_RGB32);

      // prepare the work items in individual structures
      WorkItem *w = new WorkItem[Xparts * Yparts];
      for (int i = 0; i < Xparts; i++)
        for (int j = 0; j < Yparts; j++)
          {
            int idx = j * Xparts + i;

            w[idx].upperX = upperCornerX + i * partXSpan;
            w[idx].upperY = upperCornerY - j * partYSpan;
            w[idx].lowerX = upperCornerX + (i + 1) * partXSpan;
            w[idx].lowerY = upperCornerY - (j + 1) * partYSpan;

            w[idx].imageX = i * workItemPixelsPerSide;
            w[idx].imageY = j * workItemPixelsPerSide;
            w[idx].pixelsX = workItemPixelsPerSide;
            w[idx].pixelsY = workItemPixelsPerSide;
          }

      // now distribute the work item to the worker nodes
      int *assignedPart = new int[N];   // keep track of what its worker is assigned
      int *imgPart = new int[workItemPixelsPerSide * workItemPixelsPerSide];
      for (int i = 0; i < Xparts * Yparts; i++)
        {
          MPI_Recv (imgPart, workItemPixelsPerSide * workItemPixelsPerSide, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
          int workerID = status.MPI_SOURCE;
          int tag = status.MPI_TAG;
          int widx = assignedPart[workerID];
          assignedPart[workerID] = i;
          MPI_Isend (&(w[i]), 1, workItemType, workerID, WORKITEMTAG, MPI_COMM_WORLD, &request); 
          if (tag == RESULTTAG)
            {
              savePixels (img, imgPart, w[widx].imageX, w[widx].imageY, workItemPixelsPerSide, workItemPixelsPerSide);
            }
        }

      // now send termination messages
      for (int i = 1; i < N; i++)
        {
          MPI_Recv (imgPart, workItemPixelsPerSide * workItemPixelsPerSide, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

          int workerID = status.MPI_SOURCE;
          int tag = status.MPI_TAG;

          if (tag == RESULTTAG)
            {
              int widx = assignedPart[workerID];
              savePixels (img, imgPart, w[widx].imageX, w[widx].imageY, workItemPixelsPerSide, workItemPixelsPerSide);
            }
          assignedPart[workerID] = -1;
          MPI_Isend (NULL, 0, workItemType, workerID, ENDTAG, MPI_COMM_WORLD, &request);
        }

      img->save ("mandel.png", "PNG", 0); // save the resulting image

      delete[]w;
      delete[]assignedPart;
      delete[]imgPart;

      end_time = MPI_Wtime ();
      cout << "Total time : " << end_time - start_time << endl;
    }
  else      // worker code
    {
      MandelCompute c;
      MPI_Send (NULL, 0, MPI_INT, 0, NULLRESULTTAG, MPI_COMM_WORLD);    // establish communication with master
      while (1)
        {
          WorkItem w;
          MPI_Recv (&w, 1, workItemType, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); // get a new work item
          int tag = status.MPI_TAG;
          if (tag == ENDTAG)
            break;

          c.init (&w);
          int *res = c.compute ();
          MPI_Send (res, w.pixelsX * w.pixelsY, MPI_INT, 0, RESULTTAG, MPI_COMM_WORLD); // return the results
        }
    }

  MPI_Finalize ();
  return 0;
}
