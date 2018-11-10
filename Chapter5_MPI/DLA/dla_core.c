/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Core library for simulating DLA phenomena
                 Basic functions are:
                    - dla_evolve : is based on a 2D array that is populated by particles
                    - dla_evolve_plist : is based on a separate list for still mobile particles. The crystal formation is still in the 2D array 
 To build use  : make
 ============================================================================
 */
#include <stdlib.h>
#include "dla_core.h"

/*------------------------------------------------
 * Checks the presence of a structure in neighboring cells 
 * pic points to 2D array holding data, arranged in cols
 * columns. Because the array is two columns and two rows 
 * wider than necessary, there is no need to check for 
 * boundary values of x and y.
 */
int check_proxim (int *pic, int cols, int x, int y)
{
  int *row0, *row1, *row2;
  row0 = pic+(y - 1) * cols + x - 1;
  row1 = row0 + cols;
  row2 = row1 + cols;
  if (*row0 < 0 || *(row0+1) < 0 || *(row0+2) < 0 ||
      *row1 < 0 || *(row1+1) < 0 || *(row1+2) < 0 ||
      *row2 < 0 || *(row2+1) < 0 || *(row2+2) < 0   )
    return (-1);
  else
    return (1);
}
// Unoptimized but easier to understand variation:
// int check_proxim_ (int *pic, int cols, int x, int y)
// {
//   if (pic[(y - 1) * cols + x] < 0 ||
//       pic[(y + 1) * cols + x] < 0 ||
//       pic[(y - 1) * cols + (x - 1)] < 0 ||
//       pic[(y + 1) * cols + (x - 1)] < 0 ||
//       pic[(y - 1) * cols + (x + 1)] < 0 ||
//       pic[(y + 1) * cols + (x + 1)] < 0 ||
//       pic[y * cols + (x - 1)] < 0 || pic[y * cols + (x + 1)] < 0)
//     return (-1);
//   else
//     return (1);
// }

/*------------------------------------------------*/
/* Returns -1,0 and 1 with equal probability */
inline int three_way ()
{ 
  return (random() % 3) -1;
//  long aux = random ();
//   if (aux < RAND_MAX / 3)
//     return (-1);
//   else if (aux < (RAND_MAX / 3) << 1)
//     return (0);
//   else
//     return (1);
}

/*------------------------------------------------*/
/* Initializes the 2D array for the simulation */
void dla_init (int *pic, int rows, int cols, int particles, int init_seed)
{
  int i, j, x, y;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      pic[i * cols + j] = 0;

  for (i = 0; i < particles; i++)	/* generate initial particle placement */
    {
      x = random () % (cols-2) + 1;	// counting starts from 1
      y = random () % (rows-2) + 1;
      if ((y == rows / 2 + 1) && (x == cols / 2 + 1))	// repeat if true
	i--;
      else
	pic[y * cols + x]++;
    }

  if(init_seed)
      pic[(rows / 2 ) * cols + (cols / 2) ] = -1;	/* place initial seed */
}
/*------------------------------------------------*/
/* Initializes the 2D array and array of particles for the simulation */
void dla_init_plist (int *pic, int rows, int cols, PartStr *p, int particles, int init_seed)
{
  int i, j, x, y;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      pic[i * cols + j] = 0;

  if(init_seed)
      pic[(rows / 2 ) * cols + (cols / 2)] = -1;	/* place initial seed */

  for (i = 0; i < particles; i++)	/* generate initial particle placement */
    {
      x = random () % (cols-2) + 1;	// counting starts from 1
      y = random () % (rows-2) + 1;
      if ((y == rows / 2 + 1) && (x == cols / 2 + 1))	// repeat if true
	i--;
      else
      {
	p[i].x = x;
	p[i].y = y;
      }
    }
}

/*------------------------------------------------
 * Single step evolution of the simulation.
 * 
 * The cell values represent:
 * 	0: empty space
 * 	>0 : multiple particles
 * 	<0 : crystal
 * 
 * Returns the address of the structure holding the last update */
int *dla_evolve (int *pic, int *pic2, int rows, int cols)
{
  int x, y, k;

  // prepare array to hold new state
  for (y = 1; y < rows-1; y++)
    for (x = 1; x < cols-1; x++)
      pic2[y * cols + x] = pic[y * cols + x] > 0 ? 0 : pic[y * cols + x];

  for (y = 1; y < rows - 1; y++)
    for (x = 1; x < cols - 1; x++)
      for (k = 0; k < pic[y * cols + x]; k++)
	{
	  int new_x = x + three_way ();
	  if (new_x == 0)
	    new_x = 1;
	  else if (new_x == cols - 1)
	    new_x = cols - 2;

	  int new_y = y + three_way ();
	  if (new_y == 0)
	    new_y = 1;
	  else if (new_y == rows - 1)
	    new_y = rows - 2;

	  if (pic2[new_y * cols + new_x] > 0)	// steps into empty space
	    pic2[new_y * cols + new_x]++;
	  else if (pic2[new_y * cols + new_x] == 0)	// steps into unchecked space
	    {
	      pic2[new_y * cols + new_x] =
		check_proxim (pic2, cols, new_x, new_y);
	    }
	  /* if pic2[new_y * cols + new_x] <0 the particle steps into space that will be part
	   * of the structure on the next iteration. In this case nothing is done and
	   * the particle is effectively "lost", which helps calculation simplicity */
	}

  return pic2;
}
/*------------------------------------------------
 * Single step evolution of the simulation.
 * 
 * The cell values represent:
 * 	0  : empty space
 * 	<0 : crystal  -- for consistency with alternative formulation
 * 
 * Particles are held in a separate array.
 */
void dla_evolve_plist (int *pic, int rows, int cols, PartStr *p, int *particles, PartStr *changes, int *numChanges)
{
  int i;
  *numChanges=0;
  
  for (i = 0; i < *particles; i++)
	{
	  int new_x = p[i].x + three_way ();
	  if (new_x == 0)  // bounce off boundaries
	    new_x = 1;
	  else if (new_x == cols - 1)
	    new_x = cols - 2;

	  int new_y = p[i].y + three_way ();
	  if (new_y == 0)  // bounce off boundaries
	    new_y = 1;
	  else if (new_y == rows - 1)
	    new_y = rows - 2;

	  if (pic[new_y * cols + new_x] == 0)	// steps into empty space
	    {
	      int turnCrystal = check_proxim (pic, cols, new_x, new_y);
	      if(turnCrystal<0)
	      {
		 // The application of changes can take place here in a sequential program
		 // but this affects the behavior of the code. For consistency with the 
		 // parallel version, a separate function is used for this purpose
                 // pic[new_y * cols + new_x] = -1;
                 
		 // record crystal change 
		 changes[*numChanges].x = new_x;
		 changes[*numChanges].y = new_y;
		 (*numChanges) ++;
		 // erase particle from list
		 p[i] = p[(*particles) - 1];
		 i--;
		 (*particles)--;
	      }
	      else  // change position to particle
	      {
		p[i].x=new_x;
		p[i].y=new_y;
	      }
	    }
	}
}
/*------------------------------------------------*/
void apply_changes(int *pic, int rows, int cols, PartStr *changes, int numChanges)
{
  int i;
  for(i=0;i<numChanges;i++)
     pic[changes[i].y * cols + changes[i].x] = -1;    
}