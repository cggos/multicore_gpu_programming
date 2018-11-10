/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp thrsafe_strtok.cpp -o thrsafe_strtok
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

omp_lock_t l;
//---------------------------------------------
void threadSafeParse (char *s, const char *delim)
{
  omp_set_lock (&l);
  char *tok;
  tok = strtok (s, delim);
  while (tok)
    {
      printf ("Thread %i : %s\n", omp_get_thread_num (), tok);
      tok = strtok (NULL, delim);
    }

  omp_unset_lock (&l);
}

//---------------------------------------------
int main (int argc, char *argv[])
{
  if (argc != 4)
    {
      fprintf (stderr, "Usage: %s string1 string2 delim\n", argv[0]);
      exit (EXIT_FAILURE);
    }
  char *str1 = argv[1], *str2 = argv[2], *delim = argv[3];

#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp task
      threadSafeParse (str1, delim);

#pragma omp task
      threadSafeParse (str2, delim);
    }
  }

  exit (EXIT_SUCCESS);
}
