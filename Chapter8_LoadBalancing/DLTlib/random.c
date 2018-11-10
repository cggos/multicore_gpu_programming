/*
    Part of the DLTlib library
    Copyright (C) 2014, Gerassimos Barlas
    Contact : gerassimos.barlas@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses
*/

/*-- Random number generation routines taken from Numerical Recipes in C --*/
#include "/papers/cpp_lib/random.h"
#include <stdlib.h>
#include <math.h>

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2E-7
#define RNMX (1.0-EPS)

#define CUMU_PROB_ERROR  0.000001 /* cumulative probability error */

float ran2(long *idum)
{
  int j;
  long k;
  static long idum2=123456789;
  static long iy=0;
  static long iv[NTAB];
  float temp;

  if(*idum < 0)
    {
      if(-(*idum)<1) *idum=1;
      else *idum= -(*idum);
      idum2=(*idum);
      for(j=NTAB+7;j>=0;j--)
        {
          k=(*idum)/IQ1;
          *idum=IA1*(*idum-k*IQ1)-k*IR1;
          if(*idum<0) *idum+=IM1;
          if(j<NTAB) iv[j]=*idum;
        }
      iy=iv[0];
    }
  k=(*idum)/IQ1;
  *idum=IA1*(*idum-k*IQ1)-k*IR1;
  if(*idum<0) *idum+=IM1;
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;
  if(idum2<0) idum2+=IM2;
  j=iy/NDIV;
  iy=iv[j]-idum2;
  iv[j]=*idum;
  if(iy<1) iy+=IMM1;
  if((temp=AM*iy)>RNMX) return RNMX;
  else return(temp);
}

#undef IM1 
#undef IM2 
#undef AM 
#undef IMM1
#undef IA1 
#undef IA2 
#undef IQ1 
#undef IQ2 
#undef IR1 
#undef IR2 
#undef NTAB
#undef NDIV
#undef EPS 
#undef RNMX
/*----------------------------------------------------------------*/
float gasdev(long *idum)
{
  static int iset=0;
  static float gset;
  float fac,r,v1,v2;

  if  (iset == 0) {
       do {
            v1=2.0*ran2(idum)-1.0;
            v2=2.0*ran2(idum)-1.0;
            r=v1*v1+v2*v2;
          } while (r >= 1.0 || r == 0.0);
       fac=sqrt(-2.0*log(r)/r);
       gset=v1*fac;
       iset=1;
       return v2*fac;
   } else {
       iset=0;
       return gset;
   }
}
/*---------------------------------------------------------*/
/* An alternative normal distribution random generator based on the
 * Central Limit Theorem */
float gauss()
{
  long temp=0;
  int i;

  for(i=0;i<16;i++)
    temp+=rand();
  return(1.0*temp/RAND_MAX-8);
}
/*---------------------------------------------------------*/
/* Given a value for the cumulative probability it returns the value z for
/* which P(X <= z) == cumulative  */
double find_z(double cumulative)
{
        double z1,z2,z3;
        double cum1,cum2,cum3;
        
        z1=-5;
        z3=5;
        cum1=0;
        cum3=1;
        z2 = (z3-z1)/2 +z1;
        cum2 =  0.5*(1+ erf(z2 /M_SQRT2));
        while( fabs(cum2 - cumulative) > CUMU_PROB_ERROR)
        {
                if(cum2 < cumulative)
                {
                        z1=z2;
                        cum1=cum2;              
                }
                else
                {
                        z3=z2;
                        cum3=cum2;
                }
        z2 = (z3-z1)/2 +z1;
        cum2 =  0.5*(1+ erf(z2 /M_SQRT2));      
        }
        return(z2);
}

