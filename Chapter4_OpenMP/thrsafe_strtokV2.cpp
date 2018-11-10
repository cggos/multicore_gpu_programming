/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp thrsafe_strtokV2.cpp -o thrsafe_strtokV2
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>


char *strtokV2 (char *s, const char *delim, char **aux)
{
  int idx1 = 0, idx2 = -1;
  char needle[2] = { 0 };
  int i;
  char *temp = s;
  if (s == NULL)
    temp = *aux;

  // iterate over all characters of the input string
  for (i = 0; temp[i]; i++)
    {
     // printf("%i %i %c\n", i, omp_get_thread_num (), temp[i]);
      needle[0] = temp[i];
      // check if a character matches a delimiter
      if (strstr (delim, needle) != NULL)       // strstr is reentrant
        {
          idx1 = idx2 + 1;      // get the index boundaries of the token
          idx2 = i;
          if (idx1 != idx2)     // is it a token or a delimiter following another?
            {
              temp[i] = 0;
              *aux = temp + i + 1;
              return temp+idx1;
            }
        }
    }

  // repeat checks for the token preceding the end of the string
  idx1 = idx2 + 1;
  idx2 = i;
  if (idx1 != idx2)
    {
      *aux = temp + i;
      return temp+idx1;
    }
  else
    return NULL;
}

//---------------------------------------------
void threadSafeParse (char *s, const char *delim)
{
  char *state;
  char *tok;

  tok = strtokV2 (s, delim, &state);
  while (tok)
    {
      printf ("Thread %i : %s\n", omp_get_thread_num (), tok);
      tok = strtokV2 (NULL, delim, &state);
    }
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
