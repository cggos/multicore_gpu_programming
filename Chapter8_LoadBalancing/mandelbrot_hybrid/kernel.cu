#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define CUDA_CHECK_RETURN(value) { \
        cudaError_t _m_cudaStat = value;\
        if (_m_cudaStat != cudaSuccess) {\
                fprintf(stderr, "Error %s at line %d in file %s\n",\
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
                exit(1);\
        } }

static const int BLOCK_SIDE = 16;       // size of 2D block of threads

//************************************************************
__device__ int diverge (double cx, double cy, int MAXITER)
{
  int iter = 0;
  double vx = cx, vy = cy, tx, ty;
  while (iter < MAXITER && (vx * vx + vy * vy) < 4)
    {
      tx = vx * vx - vy * vy + cx;
      ty = 2 * vx * vy + cy;
      vx = tx;
      vy = ty;
      iter++;
    }
  return iter;
}


//************************************************************
__global__ void mandelKernel (unsigned *d_res, double upperX, double upperY, double stepX, double stepY, int resX, int resY, int pitch, int MAXITER)
{
  int myX, myY;
  myX = blockIdx.x * blockDim.x + threadIdx.x;
  myY = blockIdx.y * blockDim.y + threadIdx.y;
  if (myX >= resX || myY >= resY)
    return;

  double tempx, tempy;
  tempx = upperX + myX * stepX;
  tempy = upperY - myY * stepY;
  int color = diverge (tempx, tempy, MAXITER);
  d_res[myY * pitch/sizeof(int) + myX] = color;
}

//************************************************************
int maxResX = 0;
int maxResY = 0;
int pitch = 0;
unsigned int *h_res;
unsigned int *d_res;
//************************************************************
extern "C" void CUDAmemCleanup ()
{
  CUDA_CHECK_RETURN (cudaFreeHost (h_res));
  CUDA_CHECK_RETURN (cudaFree (d_res));
}

//************************************************************
extern "C" unsigned int *CUDAmemSetup (int maxResX, int maxResY)
{
  CUDA_CHECK_RETURN (cudaMallocPitch ((void **) &d_res, (size_t *) & pitch, maxResX * sizeof (unsigned), maxResY));
  CUDA_CHECK_RETURN (cudaHostAlloc (&h_res, maxResY * pitch, cudaHostAllocMapped));
  return h_res;
}


//************************************************************
// Host front-end function that allocates the memory and launches the GPU kernel
extern "C" void hostFE (double upperX, double upperY, double lowerX, double lowerY, int resX, int resY, unsigned int **pixels, int *currpitch, int MAXITER)
{
  int blocksX, blocksY;
  blocksX = (int) ceil (resX * 1.0/ BLOCK_SIDE);
  blocksY = (int) ceil (resY * 1.0/ BLOCK_SIDE);
  dim3 block (BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid (blocksX, blocksY);


  int ptc = 32;
  while (ptc < resX * sizeof (unsigned))
    ptc += 32;

  double stepX = (lowerX - upperX) / resX;
  double stepY = (upperY - lowerY) / resY;

  // launch GPU kernel
  mandelKernel <<< grid, block >>> (d_res, upperX, upperY, stepX, stepY, resX, resY, ptc, MAXITER);

  // get the results
  CUDA_CHECK_RETURN (cudaMemcpy (h_res, d_res, resY * ptc, cudaMemcpyDeviceToHost));
  *pixels = h_res;
  *currpitch = ptc;
}
