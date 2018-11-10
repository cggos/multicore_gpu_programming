/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : DLA sequential program
 To build use  : make
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "dla_core.h"

#define PIC_SIZE 100
#define PARTICLES 500
#define MAX_ITER 10000

/*------------------------------------------------*/
int main (int argc, char **argv)
{
  int cols, rows, iter, particles, x, y;
  int *pic, *pic2, *aux;

  if (argc < 2)  // use default values if user does not specify anything
    {
      cols = PIC_SIZE + 2;
      rows = PIC_SIZE + 2;
      iter = MAX_ITER;
      particles = PARTICLES;
    }
  else
    {
      cols = atoi (argv[1]) + 2;
      rows = atoi (argv[2]) + 2;
      particles = atoi (argv[3]);
      iter = atoi (argv[4]);
    }

  // initialize the random number generator
  srand(time(0));
    
  pic = (int *) malloc (sizeof (int) * cols * rows);
  pic2 = (int *) malloc (sizeof (int) * cols * rows);
  dla_init (pic, rows, cols, particles, 1);
  while (iter--)
    {
      aux = dla_evolve (pic, pic2, rows, cols);
      pic2 = pic;
      pic = aux;
    }

  /* Print to stdout a PBM picture of the simulation space */
  printf ("P1\n%i %i\n", cols-2, rows-2);

  for (y = 1; y < rows-1; y++)
    {
      for (x = 1; x < cols-1; x++)
	{
	  if (pic[y * cols + x] < 0)
	    printf ("1 ");
	  else
	    printf ("0 ");
	}
      printf ("\n");
    }
  free (pic);
  free (pic2);
  return (0);
}
