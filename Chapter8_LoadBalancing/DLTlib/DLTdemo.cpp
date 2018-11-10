/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Demo of DLTlib use
 To build use  : g++ DLTdemo.cpp -o DLTdemo -lglpk 
 ============================================================================
 */
#include <time.h>
#include <stdio.h>
#include <iostream>

using namespace std;

//------------------------------------------------------------------------
// DLTlib specific definitions that need to be used by the library.
// Control of the seed allows manipulation of the output of pseudo-random generated structures
// in case repetitive tests are needed.
long global_random_seed;

#include "dltlib.cpp"
//------------------------------------------------------------------------

int main ()
{
  double p_cpu = 0.01;
  double p_gpu = 0.005;
  double l = 0.01;
  long int L = 1000;
  int M = 2;                    // number of installments

  // STEP 1
  Network platform;             // object representing parallel platform

  // STEP 2
  // insert one-by-one the nodes that make up the machine
  // LON stands for Load Originating Node. It can be considered to be 
  // equivalent to the file server as it does not participate in the computation
  platform.InsertNode ((char *) "LON", p_cpu, 0, (char *) NULL, l, true);
  platform.InsertNode ((char *) "GPU", p_gpu, 0, (char *) "LON", l, true);
  platform.InsertNode ((char *) "CPU0", p_cpu, 0, (char *) "LON", l, true);
  platform.InsertNode ((char *) "CPU1", p_cpu, 0, (char *) "LON", l, true);
  platform.InsertNode ((char *) "CPU2", p_cpu, 0, (char *) "LON", l, true);

  // STEP 3
  // Solve the partitioning problem for 1-port, block-type computation and M installments 
  double execTime = platform.SolveImageQuery_NInst (L, 1, 0, M);

  // print out the results, if the solution is valid
  if (platform.valid == 1)
    {
      cout << "Predicted execution time: " << execTime << endl;

      // STEP 4
      // Compute nodes are stored in a public linked-list that allows 
      // rearrangement to the order of distribution and collection
      cout << "Solution in terms of load percent :\n";
      Node *h = platform.head;
      while (h != NULL)
        {
          // For a single installment case, the following statement should be used
          //cout << h->name << " " << h->part << endl;

          cout << h->name;
          for (int i = 0; i < M; i++)
            cout << "\t" << h->mi_part[i];      // array mi_part holds the parts for each installment
          cout << endl;
          h = h->next_n;
        }

      cout << "Solution in terms of images :\n";
      h = platform.head;
      while (h != NULL)
        {
          cout << h->name;
          for (int i = 0; i < M; i++)
            cout << "\t" << h->mi_part[i] * L;
          cout << endl;
          h = h->next_n;
        }
    }
  else
    cout << "Solution could not be found\n";
  return 0;
}
