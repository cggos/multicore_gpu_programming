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
#include <QImage>
#include <QRgb>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

static const int MAXITER = 255;
static const int THR_BLK_X = 8; // pixels per thread, x-axis
static const int THR_BLK_Y = 8; // pixels per thread, y-axis
static const int BLOCK_SIDE = 16;       // size of 2D block of threads

//************************************************************
__device__ int diverge (double cx, double cy)
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
__global__ void mandelKernel (unsigned char *d_res, double upperX, double upperY, double stepX, double stepY, int resX, int resY, int pitch)
{
  int myX, myY;
  myX = (blockIdx.x * blockDim.x + threadIdx.x) * THR_BLK_X;
  myY = (blockIdx.y * blockDim.y + threadIdx.y) * THR_BLK_Y;

  int i, j;
  for (i = myX; i < THR_BLK_X + myX; i++)
    for (j = myY; j < THR_BLK_Y + myY; j++)
      {
        // check for "outside" pixels
        if (i >= resX || j >= resY)
          continue;

        double tempx, tempy;
        tempx = upperX + i * stepX;
        tempy = upperY - j * stepY;

        int color = diverge (tempx, tempy);
        d_res[j * pitch + i] = color;
      }
}

//************************************************************
// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (double upperX, double upperY, double lowerX, double lowerY, QImage * img, int resX, int resY)
{
  int blocksX, blocksY;
  blocksX = (int) ceil (resX *1.0/ (BLOCK_SIDE * THR_BLK_X));
  blocksY = (int) ceil (resY *1.0/ (BLOCK_SIDE * THR_BLK_Y));
  dim3 block (BLOCK_SIDE, BLOCK_SIDE);
  dim3 grid (blocksX, blocksY);

  int pitch;

  unsigned char *h_res;
  unsigned char *d_res;

  CUDA_CHECK_RETURN (cudaMallocPitch ((void **) &d_res, (size_t *) & pitch, resX, resY));
  CUDA_CHECK_RETURN (cudaHostAlloc (&h_res, resY * pitch, cudaHostAllocMapped));

  double stepX = (lowerX - upperX) / resX;
  double stepY = (upperY - lowerY) / resY;

  // launch GPU kernel
  mandelKernel <<< grid, block >>> (d_res, upperX, upperY, stepX, stepY, resX, resY, pitch);

  // wait for GPU to finish
  CUDA_CHECK_RETURN (cudaThreadSynchronize ());

  // get the results
  CUDA_CHECK_RETURN (cudaMemcpy (h_res, d_res, resY * pitch, cudaMemcpyDeviceToHost));

  //copy results into QImage object   
  for (int j = 0; j < resY; j++)
    for (int i = 0; i < resX; i++)
      {
        int color = h_res[j * pitch + i];
        img->setPixel (i, j, qRgb (256 - color, 256 - color, 256 - color));
      }

  // clean-up allocated memory
  CUDA_CHECK_RETURN (cudaFreeHost (h_res));
  CUDA_CHECK_RETURN (cudaFree (d_res));

}
