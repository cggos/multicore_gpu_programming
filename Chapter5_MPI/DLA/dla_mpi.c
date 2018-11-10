/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : DLA MPI program
                 Particles are evenly distributed among nodes/processes
 To build use  : make
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "dla_core.h"
#include <mpi.h>

#define PIC_SIZE 100
#define PARTICLES 500
#define MAX_ITER 10000

/*------------------------------------------------*/
int main (int argc, char **argv)
{
  int cols, rows, iter, particles, x, y;
  int *pic;
  PartStr *p, *changes, *totalChanges;
  int rank, num, i, numChanges, numTotalChanges;
  int *changesPerNode, *buffDispl;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &num);
  
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
  srand(rank);
  // srand(time(0)); // this should be used instead if the program runs on multiple hosts

    
  int particlesPerNode = particles / num;
  if (rank == num - 1)
    particlesPerNode = particles - particlesPerNode * (num - 1);	// in case particles cannot be split evenly
// printf("%i has %i\n", rank, particlesPerNode);
  pic = (int *) malloc (sizeof (int) * cols * rows);
  p = (PartStr *) malloc (sizeof (PartStr) * particlesPerNode);
  changes = (PartStr *) malloc (sizeof (PartStr) * particlesPerNode);
  totalChanges = (PartStr *) malloc (sizeof (PartStr) * particlesPerNode);
  changesPerNode = (int *) malloc (sizeof (int) * num);
  buffDispl = (int *) malloc (sizeof (int) * num);
  assert (pic != 0 && p != 0 && changes != 0 && totalChanges != 0
	  && changesPerNode != 0);



  // MPI user type declaration
  int lengths[2] = { 1, 1 };
  MPI_Datatype types[2] = { MPI_INT, MPI_INT };
  MPI_Aint add1, add2;
  MPI_Aint displ[2];
  MPI_Datatype Point;

  MPI_Address (p, &add1);
  MPI_Address (&(p[0].y), &add2);
  displ[0] = 0;
  displ[1] = add2 - add1;

  MPI_Type_struct (2, lengths, displ, types, &Point);
  MPI_Type_commit (&Point);


  dla_init_plist (pic, rows, cols, p, particlesPerNode, 1);
  while (--iter)
    {
      dla_evolve_plist (pic, rows, cols, p, &particlesPerNode, changes, &numChanges);      
//       printf("%i changed %i on iter %i : ",rank, numChanges, iter);
//       for(i=0;i<numChanges;i++) printf("(%i, %i) ", changes[i].x, changes[i].y);
//       printf("\n");
      
      //exchange information with other nodes
      MPI_Allgather (&numChanges, 1, MPI_INT, changesPerNode, 1, MPI_INT, MPI_COMM_WORLD);
      //calculate offsets
      numTotalChanges = 0;
      for (i = 0; i < num; i++)
	{
	  buffDispl[i] = numTotalChanges;
	  numTotalChanges += changesPerNode[i];
	}
//        if(rank==0)
//        {
//  	for(i=0;i<num;i++)
//  	  printf("%i tries to send %i\n",i,changesPerNode[i]);
//  	printf("-----------\n");
//        }
      if(numTotalChanges >0)
      {
      MPI_Allgatherv (changes, numChanges, Point,
		      totalChanges, changesPerNode, buffDispl, Point,
		      MPI_COMM_WORLD);
      apply_changes (pic, rows, cols, totalChanges, numTotalChanges);

	
//       if(rank==0)
//       {
//         printf("Total changes %i : ", numTotalChanges);
//         for(i=0;i<numTotalChanges;i++) printf("(%i, %i) ", totalChanges[i].x, totalChanges[i].y);
// 	
//         printf("\n");
// 	printf("-----------\n");
//       }
      }
    }

  /* Print to stdout a PBM picture of the simulation space */
  if (rank == 0)
    {
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
    }
    
  MPI_Reduce(&particlesPerNode, &particles, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);  
  if(rank==0) fprintf(stderr, "Remaining particles %i\n", particles);
    
  free (pic);
  free (p);
  free (changes);
  free (changesPerNode);
  free (buffDispl);
  MPI_Finalize ();
  return 0;
}
