/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <ctime>

#include "rijndael.h"

using namespace std;

static const int keybits = 256;

//******************************************************************************************************
int main (int argc, char *argv[])
{
  int lSize = 0;
  FILE *f, *f2;
  unsigned char *iobuf;
  timeval timeMain;
  
  if (argc < 4)
    {
      fprintf (stderr, "Usage : %s inputfile outputfile threadsPerBlock\n", argv[0]);
      exit (1);
    }

    
  gettimeofday (&timeMain, NULL);
  double tM0 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);

  //encryption key
//  unsigned char key[32] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
  unsigned char key[32] = { '1', 0};
  u32 rk[RKLENGTH (keybits)];
  // expanded key preparation
  int nrounds = rijndaelSetupEncrypt (rk, key, keybits);

  if ((f = fopen (argv[1], "r")) == NULL)
    {
      fprintf (stderr, "Can't open %s\n", argv[1]);
      exit (EXIT_FAILURE);
    }

  if ((f2 = fopen (argv[2], "w")) == NULL)
    {
      fprintf (stderr, "Can't open %s\n", argv[2]);
      exit (EXIT_FAILURE);
    }

  int thrPerBlock = atoi (argv[3]);

  fseek (f, 0, SEEK_END);
  lSize = ftell (f);
  rewind (f);

  iobuf = new unsigned char[lSize];
  assert (iobuf != 0);
  fread (iobuf, 1, lSize, f);
  fclose (f);

  gettimeofday (&timeMain, NULL);
  double tM1 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);
  
  rijndaelEncryptFE (rk, keybits, iobuf, iobuf, lSize, thrPerBlock);
  rijndaelShutdown ();

  gettimeofday (&timeMain, NULL);
  double tM2 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);
  
  fwrite(iobuf, 1, lSize, f2);
  fclose(f2);
  delete[]iobuf;


   // print-out some timing information
  gettimeofday (&timeMain, NULL);
  double tM3 = timeMain.tv_sec + (timeMain.tv_usec / 1000000.0);
   printf ("%.9lf \t %.9lf \n", tM3-tM0, tM2-tM1);

  return 0;
}
