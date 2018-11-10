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
#include <math.h>
#include <assert.h>
#include <iostream>
using namespace std;

//******************************************************************************************************
// Returns the predicted execution time
// and the calculated parts in array part[]
double nPortPartition(double *p, double *e, long L, double l, double b, double d, double *part, int N)
{
  // temporary arrays for speeding-up the calculation
  double lacp[N];
  double sumTerms[N];
  
  lacp[0]=1.0/p[0];
  for(int i=1;i<N;i++)
    lacp[i] = 1.0/(l*2+p[i]);

  // sumTerms[0] is not utilized
  for(int i=1;i<N;i++)
    sumTerms[i] = (p[0]*e[0] - p[i]*e[i]-l*(b+d)) * lacp[i] / L;
  
  // calculate the nominator and denominator for finding part_0  
  double nomin=1, denom=0;
  for(int i=1;i<N;i++)
  {
    nomin -= sumTerms[i];
    denom += lacp[i];
  }
  denom*=p[0];
  denom++;
  
  part[0] = nomin/denom;
  
  // calculate the other parts now
  for(int i=1;i<N;i++)
    part[i] = part[0] * p[0]*lacp[i] + sumTerms[i];

//     for(int i=0;i<N;i++)
//       cerr << i << " " << part[i] << endl;
  
  // sanity check - always a good idea!
  double sum=0;
  for(int i=0;i<N;i++)
    sum += part[i];
  assert(fabs(sum -1)<0.001);
  
  // return the exec. time
  return l*(2*part[0] + b+d) + p[0]*(part[0]*L+e[0]);
}
//******************************************************************************************************
// returns the assigned load in array assign[], that are multiples of quantum (assuming L is also a multiple of quantum)
void quantize(double *part, long L, int quantum, int N, long *assign)
{
   int totAssigned=0;
   for(int i=1;i<N;i++)
   {
     // truncate the parts assigned to all workers but node 0
     assign[i] = ((long)floor(part[i]*L/quantum))*quantum;
     totAssigned += assign[i];     
   }
   // node 0 gets everything else
   assign[0] = L - totAssigned;
}
