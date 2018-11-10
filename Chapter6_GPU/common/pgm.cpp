/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "pgm.h"

using namespace std;

//-------------------------------------------------------------------
PGMImage::PGMImage(char *fname)
{
   x_dim=y_dim=num_colors=0;
   pixels=NULL;
   
   FILE *ifile;
   ifile = fopen(fname, "rb");
   if(!ifile) return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &x_dim, &y_dim, &num_colors);

   getline((char **)&buff, &temp, ifile); // eliminate CR-LF
   
   assert(x_dim >1 && y_dim >1 && num_colors >1);
   pixels = new unsigned char[x_dim * y_dim];
   fread((void *) pixels, 1, x_dim*y_dim, ifile);   
   
   fclose(ifile);
}
//-------------------------------------------------------------------
PGMImage::PGMImage(int x=100, int y=100, int col=16)
{
   num_colors = (col>1) ? col : 16;
   x_dim = (x>1) ? x : 100;
   y_dim = (y>1) ? y : 100;
   pixels = new unsigned char[x_dim * y_dim];
   memset(pixels, 0, x_dim * y_dim);
   assert(pixels);
}
//-------------------------------------------------------------------
PGMImage::~PGMImage()
{
  if(pixels != NULL)
     delete [] pixels;
  pixels = NULL;
}
//-------------------------------------------------------------------
bool PGMImage::write(char *fname)
{
   int i,j;
   FILE *ofile;
   ofile = fopen(fname, "w+t");
   if(!ofile) return 0;

   fprintf(ofile,"P5\n%i %i\n%i\n",x_dim, y_dim, num_colors);
   fwrite(pixels, 1, x_dim*y_dim, ofile);
   fclose(ofile);
   return 1;
}
