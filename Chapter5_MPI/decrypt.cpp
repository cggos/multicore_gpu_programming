/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0.1
 Last modified : December 2015
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC decrypt.cpp -o decrypt
 ============================================================================
 */

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <rpc/des_crypt.h>
#include <mpi.h>

unsigned char cipher[]={108, 245, 65, 63, 125, 200, 150, 66, 17, 170, 207, 170, 34, 31, 70, 215, 0};
char search[]=" the ";

//----------------------------------------------------------
void decrypt(long key, char *ciph, int len)
{
   // prepare key for the parity calculation. Least significant bit in all bytes should be empty
   long k = 0;
   for(int i=0;i<8;i++)
   {
     key <<= 1;
     k += (key &  (0xFE << i*8));
   }

   des_setparity((char *)&k);

   // Decrypt ciphertext
   ecb_crypt((char *)&k,(char *) ciph, len, DES_DECRYPT);
}
//----------------------------------------------------------
// Returns true if the plaintext produced containes the "search" string
bool tryKey(long key, char *ciph, int len)
{
  char temp[len+1];
  memcpy(temp, ciph, len);
  temp[len]=0;
  decrypt(key, temp, len);
  return strstr((char *)temp, search)!=NULL;  
}

//----------------------------------------------------------
int main(int argc, char **argv)
{
  int N, id;
  long upper=(1L << 56);
  long mylower, myupper;
  MPI_Status st;
  MPI_Request req;
  int flag;
  int ciphLen = strlen((char *)cipher);
  
   MPI_Init(&argc, &argv);
   
   MPI_Comm_rank(MPI_COMM_WORLD, &id);
   MPI_Comm_size(MPI_COMM_WORLD, &N);
   int range_per_node = upper / N;

   mylower = range_per_node * id;
   myupper = range_per_node * (id + 1) - 1;
   if(id == N-1)
     myupper = upper;
   
   long found = 0;
   MPI_Irecv(&found, 1, MPI_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req);
   for(int i=mylower; i<myupper && found==0;i++)
   {
     if(tryKey(i, (char *)cipher, ciphLen))
     {
       found=i;
       for(int node=0; node<N; node++)
         MPI_Send(&found, 1, MPI_LONG, node, 0, MPI_COMM_WORLD);  
       break;
     }    
   }
 
  if(id==0)
  {
    MPI_Wait (&req, &st);
    decrypt(found, (char *)cipher,ciphLen);
    printf("%li %s\n", found, cipher);
  }
   
   MPI_Finalize();
}
