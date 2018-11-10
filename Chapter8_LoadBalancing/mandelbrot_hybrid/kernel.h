#ifndef KERNEL_H_
#define KERNEL_H_

extern "C"
void hostFE (double uX, double uY, double lX, double lY, int resX, int resY, unsigned  int **pixels, int *pitch, int maxiter);
extern "C" 
unsigned  int *CUDAmemSetup(int maxResX, int maxResY);
extern "C" 
void CUDAmemCleanup();
#endif /* KERNEL_H_ */
