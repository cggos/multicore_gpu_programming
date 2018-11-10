#ifndef H__RIJNDAEL
#define H__RIJNDAEL

typedef unsigned int u32;
typedef unsigned char u8;


int rijndaelSetupEncrypt (u32 * rk, const unsigned char *key, int keybits);
int rijndaelSetupDecrypt (u32 * rk, const unsigned char *key, int keybits);

void rijndaelEncryptFE (const u32 * rk, int keybits, unsigned char *plaintext, unsigned char *ciphertext, int N, int thrPerBlock);
void rijndaelDecryptFE (const u32 * rk, int keybits, unsigned char *ciphertext, unsigned char *plaintext, int N, int thrPerBlock);

void rijndaelShutdown ();

// CPU-only functions
void rijndaelCPUEncrypt(const u32 *rk, int nrounds, const u8 plaintext[16],  u8 ciphertext[16]);
void rijndaelCPUDecrypt(const u32 *rk, int nrounds, const u8 ciphertext[16], u8 plaintext[16]);


#define KEYLENGTH(keybits) ((keybits)/8)
#define RKLENGTH(keybits)  ((keybits)/8+28)
#define NROUNDS(keybits)   ((keybits)/32+6)


#define GETU32(plaintext) (((u32)(plaintext)[0] << 24) ^ \
                    ((u32)(plaintext)[1] << 16) ^ \
                    ((u32)(plaintext)[2] <<  8) ^ \
                    ((u32)(plaintext)[3]))

#define PUTU32(ciphertext, st) { (ciphertext)[0] = (u8)((st) >> 24); \
                         (ciphertext)[1] = (u8)((st) >> 16); \
                         (ciphertext)[2] = (u8)((st) >>  8); \
                         (ciphertext)[3] = (u8)(st); }
#endif
