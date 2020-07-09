#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <omp.h>

#define DIM 1000

struct cuComplexP {
    float   r;
    float   i;

    __device__ cuComplexP( float a, float b ) : r(a), i(b) {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplexP operator*(const cuComplexP& a) {
        return cuComplexP(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplexP operator+(const cuComplexP& a) {
        return cuComplexP(r+a.r, i+a.i);
    }
};

__device__ int juliaG( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplexP c(-0.8, 0.156);
    cuComplexP a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernelP( unsigned char *ptr ) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = juliaG( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 255 * juliaValue;
    ptr[offset*4 + 2] = 255 * juliaValue;
    ptr[offset*4 + 3] = 255;
}


struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}



double kernel( unsigned char *ptr ){
    double time = omp_get_wtime();
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
    return omp_get_wtime() - time;
 }

 double kernelMP( unsigned char *ptr ){
     double time = omp_get_wtime();
     #pragma omp parallel for
     for (int y=0; y<DIM; y++) {
         for (int x=0; x<DIM; x++) {
             int offset = x + y * DIM;

             int juliaValue = julia( x, y );
             ptr[offset*4 + 0] = 255 * juliaValue;
             ptr[offset*4 + 1] = 0;
             ptr[offset*4 + 2] = 0;
             ptr[offset*4 + 3] = 255;
         }
     }
     return omp_get_wtime() - time;
  }



int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();
    printf("Serial time = %f\n", kernel( ptr ));
    printf("Parallel time = %f\n", kernelMP( ptr ));

    HANDLE_ERROR( cudaMalloc( (void**)&ptr, bitmap.image_size() ) );
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    dim3    grid(DIM,DIM);

    cudaEventRecord(begin,0);
    kernelP<<<grid,1>>>( ptr );
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    float time = 0;
	cudaEventElapsedTime(&time, begin, end);
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ptr,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );


    printf("Kernel time = %f\n", time/1000);

    HANDLE_ERROR( cudaFree( ptr ) );
    bitmap.display_and_exit();
}
