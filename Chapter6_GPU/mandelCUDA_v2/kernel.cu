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
  myX = blockIdx.x * blockDim.x + threadIdx.x;
  myY = blockIdx.y * blockDim.y + threadIdx.y;
  if (myX >= resX || myY >= resY)
    return;

  double tempx, tempy;
  tempx = upperX + myX * stepX;
  tempy = upperY - myY * stepY;
  int color = diverge (tempx, tempy);
  d_res[myY * pitch + myX] = color;
}

//************************************************************
// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (double upperX, double upperY, double lowerX, double lowerY, QImage * img, int resX, int resY)
{
  int blocksX, blocksY;
  blocksX = (int) ceil (resX / 16.0);
  blocksY = (int) ceil (resY / 16.0);
  dim3 block (16, 16);
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
