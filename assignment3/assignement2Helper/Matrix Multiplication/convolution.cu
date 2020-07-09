#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include "deviceQuery.h"


const char *imageFilename = "../data/lena_bw.pgm";
const char *refFilename   = "lena_bw_smoothed.pgm";

#define TILE_WIDTH 8
#define BLOCK_SIZE 8

unsigned char convolution_helper(const unsigned char *image, const float *kernel, size_t width, size_t kernelWidth, int x, int y) {
	int offset = round((kernelWidth-1)/2);
	float output = 0;
	for (int i = 0; i < kernelWidth; ++i) {
		for (int j = 0; j < kernelWidth; ++j) {
			int mapi = i-offset;
			int mapj = j-offset;
			if (mapi + x >= 0 && mapi + x <= width && mapj + y >= 0 && mapj + y <= width) //check if in bounds of image
				output += (int)image[x*width+y+mapi*width+mapj] * (int)kernel[i*kernelWidth+j];
		}
	}
	return (unsigned char)output;
}

void convolution_seq(const unsigned char *image, const float *kernel, unsigned char *output, size_t width, size_t kernelWidth){
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			output[i*width+j] = convolution_helper(image, kernel, width, kernelWidth, i, j);
		}
	}
}

__global__ void matrix_multiply_simple(const float *a,const float *b, float *ab, size_t width){
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  float result = 0;

  for(int k = 0; k < width; ++k){
    result += a[row*width+k] * b[k*width+col]; //dot product
  }

  ab[row*width+col] = result;
}

__global__ void matrix_multiply_tiled (const float *a, const float *b, float *ab, size_t width) {
	__shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;
	float pvalue = 0;

	for (int ph = 0; ph < width/TILE_WIDTH; ++ph) {
		ds_a[ty][tx] = a[row*width + ph*TILE_WIDTH + tx];
		ds_b[ty][tx] = b[col+(ph*TILE_WIDTH+ty)*width];
		__syncthreads();
		for (int i = 0; i < TILE_WIDTH; ++i)
			pvalue += ds_a[ty][i] * ds_b[i][tx]; //dot product
		__syncthreads();
	}

	ab[row*width+col] = pvalue;
}

//verification
int matrixEqual(  float *matrixA, float *matrixB, int m, int n ){
  int bad = 0;
  for ( int y = 0; y < m && !bad ; y++ )
    for ( int x = 0; x < n && !bad ; x++ ){
      if ( abs(matrixA[y*n+x] - matrixB[y*n+x]) > 1e-4 ){
        bad++;
      }
    }
  return !bad;
}

int main(void){
	QueryDevice();
	std::cout << std::endl;

	unsigned int* width = (unsigned int*)malloc(sizeof(int));
	unsigned int* height = (unsigned int*)malloc(sizeof(int));
	unsigned int* channels = (unsigned int*)malloc(sizeof(int));
	unsigned char* data;

	__loadPPM(imageFilename, &data, width, height, channels);

	unsigned char* output = (unsigned char *) malloc(sizeof(unsigned char) * (*width) * (*height));

	memset(output, sizeof(unsigned char) * (*width) * (*height), 0);

	float* kernel = (float*) malloc(sizeof(float) * 3 * 3)

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			kernel[i*3+j] = (float)(1/9);
		}
	}

	convolution_seq(data, kernel, output, width, 3);

	__savePPM(refFilename, output, *width, *height, *channels);

	for (int o=0; o<1;++o) {
		// create a large workload so we can easily measure the
		// performance difference of both implementations
		// note that n measures the width of the matrix, not the number of total elements
		const size_t n = 1<<(7+o);
		//const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
		const dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
		const dim3 num_blocks(n / block_size.x, n / block_size.y);
		std::cout << "Matrix size: 2^" << (7+o) << "x2^" << (7+o) << std::endl;
		std::cout << "Tile-size: " << TILE_WIDTH << "x" << TILE_WIDTH << ", Block-size: " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
		// generate random input on the host
		float *h_a, *h_b, *h_s, *h_res;
		//std::vector<float> h_a(n*n), h_b(n*n), h_c(n*n);
		h_a = (float *)malloc(sizeof(float) * n * n);
		h_b = (float *)malloc(sizeof(float) * n * n);
		h_s = (float *)malloc(sizeof(float) * n * n);
		h_res = (float*)malloc(sizeof(float) * n * n);

		for(int i = 0; i < n*n; ++i){
		h_a[i] = static_cast<float>(rand()) / RAND_MAX;
		h_b[i] = static_cast<float>(rand()) / RAND_MAX;
		}

		// allocate storage for the device
		float *d_a = 0, *d_b = 0, *d_c = 0;
		cudaMalloc((void**)&d_a, sizeof(float) * n * n);
		cudaMalloc((void**)&d_b, sizeof(float) * n * n);
		cudaMalloc((void**)&d_c, sizeof(float) * n * n);

		// copy input to the device
		cudaMemcpy(d_a, h_a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

		// time the kernel launches using CUDA events
		cudaEvent_t launch_begin, launch_end;
		cudaEventCreate(&launch_begin);
		cudaEventCreate(&launch_end);
		//time many sequential run and take the average
		size_t num_launches = 4;
		double average_seq_time;
		struct timespec start, end;
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
		perror( "clock gettime" );
		exit( EXIT_FAILURE );
		}
		for(int i = 0; i < num_launches; i++){
		  //matrix_multiply_seq(h_a, h_b, h_s, n);
		}

		if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
		  perror( "clock gettime" );
		  exit( EXIT_FAILURE );
		}
		//compute the time in s
		average_seq_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;
		//take the average
		average_seq_time /= num_launches;
		std::cout << "Average sequential time: " << average_seq_time << "s" << std::endl;
		// launch a single "warm-up" kernel
		matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
		cudaMemcpy(h_res, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
		int equal = matrixEqual(h_res, h_s, n, n);
		if(!equal) {
			return 0;
		}
		// time many kernel launches and take the average time
		float average_simple_time = 0;
		for(int i = 0; i < num_launches; ++i){
		// record a CUDA event immediately before and after the kernel launch
		cudaEventRecord(launch_begin,0);
		matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
		cudaEventRecord(launch_end,0);
		cudaEventSynchronize(launch_end);
		// measure the time spent in the kernel
		float time = 0;
		cudaEventElapsedTime(&time, launch_begin, launch_end);
		average_simple_time += time;
		}
		average_simple_time /= num_launches;
		std::cout << "Average global kernel time: " << average_simple_time << "ms" << std::endl;
		const dim3 num_blocks_tiled(n / TILE_WIDTH, n / TILE_WIDTH);
		const dim3 block_size_tiled(TILE_WIDTH,TILE_WIDTH);
		cudaMemcpy(d_c, 0, sizeof(float), cudaMemcpyHostToDevice);
		matrix_multiply_tiled<<<num_blocks_tiled,block_size_tiled>>>(d_a, d_b, d_c, n);
		cudaMemcpy(h_res, d_c, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
		equal = matrixEqual(h_res, h_s, n, n);
		if(!equal) {
			return 0;
		}
		// time many kernel launches and take the average time
		float average_tiled_time = 0;
		for(int i = 0; i < num_launches; ++i){
			cudaEventRecord(launch_begin,0);
			matrix_multiply_tiled<<<num_blocks_tiled,block_size_tiled>>>(d_a, d_b, d_c, n);
			cudaEventRecord(launch_end,0);
			cudaEventSynchronize(launch_end);
			float time = 0;
			cudaEventElapsedTime(&time, launch_begin, launch_end);
			average_tiled_time += time;
		}
		average_tiled_time /= num_launches;
		std::cout << "Average tiled kernel time: " << average_tiled_time << "ms" << std::endl;

		float num_ops=2 * n * n * n;
		float seq_throughput = num_ops / average_seq_time / 1000000000.0f;
		float simple_throughput = num_ops / (average_simple_time / 1000.0f) / 1000000000.0f;
		float tiled_throughput = num_ops / (average_tiled_time / 1000.0f) / 1000000000.0f;

		std::cout << "Throughput of sequential implementation: " << seq_throughput << " GFLOPS" << std::endl;
		std::cout << "Throughput of global kernel: " << simple_throughput << " GFLOPS" << std::endl;
		std::cout << "Throughput of tiled kernel: " << tiled_throughput << " GFLOPS" << std::endl;
		std::cout << "Performance improvement: global over seqential " << simple_throughput / seq_throughput << "x" << std::endl;
		std::cout << "Performance speed-up: tiled over seqential " << tiled_throughput / seq_throughput << "x" << std::endl;
		std::cout << "Performance speed-up: tiled over global " << tiled_throughput / simple_throughput << "x" << std::endl;
		std::cout << std::endl;
		// destroy the CUDA events
		cudaEventDestroy(launch_begin);
		cudaEventDestroy(launch_end);

		// deallocate device memory
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(h_a);
		free(h_b);
		free(h_s);
		free(h_res);
	}
	return 0;
}
