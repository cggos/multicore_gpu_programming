#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <QThread>
#include "workqueue.h"
#include "mandelframe.h"
#include "mandelregion.h"
#include "kernel.h"


//************************************************************
class CalcThr:public QThread
{
private:
  WorkQueue * que;
  bool isGPU;

public:
    CalcThr (WorkQueue * q, bool gpu):que (q), isGPU (gpu)
  {
  }
  void run ();
};

void CalcThr::run ()
{
  MandelRegion *t;
  while ((t = que->extract ()) != NULL)
    {
      t->examine (*que, isGPU);
      delete t;
    }
}

//************************************************************
// Expects an input file with the following data:
//  numframes resolutionX resolutionY imageFilePrefix.
//  upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; for first frame
//  upperCornerX upperCornerY lowerCornerX lowerCornerY maxIterations  ; for last frame
// 
// Command-line parameters:  spec_file numThr GPUenable diffThreshold pixelThreshold
//              spec_file : the file holding the parameters mentioned above
//              numThr : number of threads (optional, defaults to the number of cores)
//              GPUenable : 0/1, 1(default) enables the GPU code (optional)
//              diffThreshold pixelThreshold : optional thresholds for frame partitioning heuristics
int main (int argc, char *argv[])
{
  int numframes, resolutionX, resolutionY;
  char imageFilePrefix[MAXFNAME - 8];
  double upperCornerX[2], upperCornerY[2], lowerCornerX[2], lowerCornerY[2];
  int maxIterations[2];
  double diffT = 0.5;
  int pixT = 32768;

  if (argc < 2)
    {
      cerr << "Usage : " << argv[0] << "spec_file numThr GPUenable\n";
      exit (1);
    }

  int numThreads = sysconf (_SC_NPROCESSORS_ONLN);
  if (argc > 2)
    numThreads = atoi (argv[2]);

  bool enableGPU = true;
  if (argc > 3)
    enableGPU = (bool) atoi (argv[3]);

  if (argc > 4)
    diffT = atof (argv[4]);

  if (argc > 5)
    pixT = atoi (argv[5]);

  ifstream fin (argv[1]);
  fin >> numframes >> resolutionX >> resolutionY;
  fin >> imageFilePrefix;
  fin >> upperCornerX[0] >> upperCornerY[0] >> lowerCornerX[0] >> lowerCornerY[0] >> maxIterations[0];
  fin >> upperCornerX[1] >> upperCornerY[1] >> lowerCornerX[1] >> lowerCornerY[1] >> maxIterations[1];
  fin.close ();

  // generate the pseudocolor map to be used for all frames
  int MAXMAXITER = max (maxIterations[0], maxIterations[1]);
  MandelRegion::initColorMapAndThrer (MAXMAXITER, diffT, pixT);

  WorkQueue workQ;

  // generate the needed frame objects and the corresponding regions
  MandelFrame **fr = new MandelFrame *[numframes];
  double uX = upperCornerX[0], uY = upperCornerY[0];
  double lX = lowerCornerX[0], lY = lowerCornerY[0];
  int iter = maxIterations[0];
  double sx1, sx2, sy1, sy2;
  int iterInc;
  sx1 = (upperCornerX[1] - upperCornerX[0]) / numframes;        // steps are a little bit smaller to avoid round-off errors causing the
  sx2 = (lowerCornerX[1] - lowerCornerX[0]) / numframes;        // last image to not render
  sy1 = (upperCornerY[1] - upperCornerY[0]) / numframes;
  sy2 = (lowerCornerY[1] - lowerCornerY[0]) / numframes;
  iterInc = (maxIterations[1] - maxIterations[0]) * 1.0 / numframes;
  char fname[MAXFNAME];
  for (int i = 0; i < numframes; i++)
    {
      sprintf (fname, "%s%04i.png", imageFilePrefix, i);
      fr[i] = new MandelFrame (uX, uY, lX, lY, resolutionX, resolutionY, fname, iter);
      workQ.append (new MandelRegion (uX, uY, lX, lY, 0, 0, resolutionX, resolutionY, fr[i]));
      uX += sx1;
      uY += sy1;
      lX += sx2;
      lY += sy2;
      iter += iterInc;
    }


  // generate the threads that will process the workload
  CalcThr **thr = new CalcThr *[numThreads];
  thr[0] = new CalcThr (&workQ, enableGPU);
  for (int i = 1; i < numThreads; i++)
    {
      thr[i] = new CalcThr (&workQ, false);
      thr[i]->start ();
    }

  // use the main thread to run one of the workers
  if (enableGPU)
    {
      CUDAmemSetup (resolutionX, resolutionY);
      thr[0]->run ();
      CUDAmemCleanup ();
    }
  else
    thr[0]->run ();

  for (int i = 1; i < numThreads; i++)
      thr[i]->wait ();

  return 0;
}
