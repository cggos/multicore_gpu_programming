/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : nvcc zip_ex.cu -o zip_ex
 ============================================================================
 */
#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

struct arbitrary_functor
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // D[i] = A[i] + B[i] * C[i];
        thrust::get<3>(t) = thrust::get<0>(t) +
	                    thrust::get<1>(t) *
	                    thrust::get<2>(t);
    }
};


int main(void)
{
    // allocate storage
    thrust::device_vector<float> A(5);
    thrust::device_vector<float> B(5);
    thrust::device_vector<float> C(5);
    thrust::device_vector<float> D(5);

    // initialize input vectors
    A[0] = 3;  B[0] = 6;  C[0] = 2; 
    A[1] = 4;  B[1] = 7;  C[1] = 5; 
    A[2] = 0;  B[2] = 2;  C[2] = 7; 
    A[3] = 8;  B[3] = 1;  C[3] = 4; 
    A[4] = 2;  B[4] = 8;  C[4] = 3; 

    // apply the transformation
    thrust::for_each(thrust::make_zip_iterator(
                       thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
                 thrust::make_zip_iterator(
		       thrust::make_tuple(A.end(),   B.end(),   C.end(),   D.end())),
                 arbitrary_functor());

    // print the output
    for(int i = 0; i < 5; i++)
        std::cout << A[i] << " + " << B[i] << " * " << C[i] << " = " << D[i] << std::endl;
}