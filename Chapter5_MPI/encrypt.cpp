/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : mpiCC encrypt.cpp -o encrypt
 ============================================================================
 */
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <rpc/des_crypt.h>

//----------------------------------------------------------
void encrypt(long key, char *ciph, int len)
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
   ecb_crypt((char *)&k,(char *) ciph, len, DES_ENCRYPT);
}
//----------------------------------------------------------
int main(int argc, char **argv)
{
  int len, origlen = strlen(argv[1]);
  if(origlen % 8 >0)
    len = ((origlen / 8)+1)*8;
  unsigned char buff[len];
  strcpy((char *)buff, argv[1]);
  long key=atol(argv[2]);
  std::cout << "Using key " << key << std::endl;
  encrypt(key, (char *)buff, len);
  std::cout<< "{";
  for(int i=0; i< len;i++)
    std::cout << (int)buff[i] << ", ";
  std::cout<< "}\n";
  return 0;
}
