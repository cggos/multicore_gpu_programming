#ifndef H__RIJNDAEL_DEVICE
#define H__RIJNDAEL_DEVICE

#define CUDA_CHECK_RETURN(value) {                                                                                      \
        cudaError_t _m_cudaStat = value;                                                                                \
        if (_m_cudaStat != cudaSuccess) {                                                                               \
                fprintf(stderr, "Error %s at line %d in file %s\n",                                     \
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);           \
                exit(1);                                                                                                                        \
        } }

__global__ void rijndaelGPUEncrypt (int nrounds, u32 * data, int N);
__global__ void rijndaelGPUDecrypt (int nrounds, u32 * data, int N);

#endif
