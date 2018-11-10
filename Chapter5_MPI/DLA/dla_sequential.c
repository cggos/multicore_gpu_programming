/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : DLA sequential program based on maintaining a list of particles
 To build use  : make
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "dla_core.h"

#define PIC_SIZE 100
#define PARTICLES 500
#define MAX_ITER 10000

/*------------------------------------------------*/
int main (int argc, char **argv)
{
  int cols, rows, iter, particles, x, y;
  int *pic;
  PartStr *p, *changes, *totalChanges;
  int i, numChanges, numTotalChanges;
  int *changesPerNode, *buffDispl;

  if (argc < 2)			// use default values if user does not specify anything
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
  p = (PartStr *) malloc (sizeof (PartStr) * particles);
  changes = (PartStr *) malloc (sizeof (PartStr) * particles);
  assert (pic != 0 && p != 0 && changes != 0);


  dla_init_plist (pic, rows, cols, p, particles, 1);
  while (--iter)
  {
    dla_evolve_plist (pic, rows, cols, p, &particles, changes, &numChanges);
    apply_changes (pic, rows, cols, changes, numChanges);
  }

  for(i=0;i<particles;i++)
    pic[p[i].y * cols + p[i].x] ++;
  
  /* Print to stdout a PBM picture of the simulation space */
  printf ("P1\n%i %i\n", cols - 2, rows - 2);

  for (y = 1; y < rows - 1; y++)
    {
      for (x = 1; x < cols - 1; x++)
	{
	  if (pic[y * cols + x] < 0)
	    printf ("1 ");
	  else
	    printf ("0 ");
	}
      printf ("\n");
    }
    
  fprintf(stderr, "Remaining %i\n", particles);    
  free (pic);
  free (p);
  free (changes);
  return 0;
}
