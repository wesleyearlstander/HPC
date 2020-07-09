#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "deviceQuery.h"

//edit this to change size of kernel if average
#define KERNEL_WIDTH 3
__constant__ float cuda_kernel[KERNEL_WIDTH * KERNEL_WIDTH];
texture <unsigned char, 2, cudaReadModeElementType> tex2;
texture <float, 2, cudaReadModeElementType> kernelTex2;
texture <unsigned char, 1, cudaReadModeElementType> tex;
texture <float, 1, cudaReadModeElementType> kernelTex;

const char* filename = "man"; //edit this to change image
const char* kernelType = "average"; //edit this to change filter (average / edge / sharpen)

int imageWidth; //automatically updated

#define TILE_WIDTH 32
#define BLOCK_SIZE 32

void convolution_seq(const unsigned char *image, const float *kernel, float *output, size_t width){
	for (int y = 0; y < width; ++y) {
		for (int x = 0; x < width; ++x) {
			int offset = KERNEL_WIDTH/2; //kernel offset with integer division
			float out = 0;
			for (int j = 0; j < KERNEL_WIDTH; ++j) { //perform convolution on a pixel
				for (int i = 0; i < KERNEL_WIDTH; ++i) {
					int mapi = i-offset;
					int mapj = j-offset;
					if (mapi + x >= 0 && mapi + x < width && mapj + y >= 0 && mapj + y < width) { //check if in bounds of image
						out += (float)image[y*width+x+mapj*width+mapi] * kernel[j*KERNEL_WIDTH+i];
					}
				}
			}
			output[y*width+x] = out;
		}
	}
}

__global__ void __launch_bounds__(1024) convolution_global(const unsigned char *image, float *output, size_t width){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			int mapi = i-offset;
			int mapj = j-offset;
			if (mapi + x >= 0 && mapi + x < width && mapj + y >= 0 && mapj + y < width) { //check if in bounds of image
				out += (float)image[y*width+x+mapj*width+mapi] * cuda_kernel[j*KERNEL_WIDTH+i];
			}
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_shared_strided(const unsigned char *image, float *output, size_t width){
	__shared__ float image_ds[(TILE_WIDTH + KERNEL_WIDTH - 1)*(TILE_WIDTH + KERNEL_WIDTH - 1)];
	int sharedWidth = (TILE_WIDTH + KERNEL_WIDTH - 1);
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int x = bx * TILE_WIDTH + tx;
	int y = by * TILE_WIDTH + ty;
	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	//load data
	int haloIndexTop = (by-1) * TILE_WIDTH + ty;
	int haloIndexBottom = (by+1) * TILE_WIDTH + ty;
	int haloIndexLeft = (bx-1) * TILE_WIDTH + tx;
	int haloIndexRight = (bx+1) * TILE_WIDTH + tx;

	if (ty >= TILE_WIDTH - offset) { //top halo pixels from left corner to right corner loaded
		if (tx >= TILE_WIDTH - offset) { //top left halo pixels loaded
			image_ds[(ty - (TILE_WIDTH - offset))*sharedWidth + tx - (TILE_WIDTH - offset)] = (haloIndexLeft<0 || haloIndexTop<0)?0:image[haloIndexTop*width+haloIndexLeft]; //top left halo pixels for tile loaded
		}
		image_ds[(ty - (TILE_WIDTH - offset))*sharedWidth+tx+offset] = (haloIndexTop<0)?0:image[haloIndexTop*width+x]; //top halo cells loaded
		if (tx < offset) {
			image_ds[(ty - (TILE_WIDTH - offset))*sharedWidth + tx + (TILE_WIDTH + offset)] = (haloIndexRight>=width || haloIndexTop<0)?0:image[haloIndexTop*width+haloIndexRight]; //top right halo pixels for tile loaded
		}
	}
	if (tx >= TILE_WIDTH - offset) { //left halo pixels loaded
		image_ds[(ty+offset)*sharedWidth + tx - (TILE_WIDTH - offset)] = (haloIndexLeft<0)?0:image[y*width+haloIndexLeft];
	}

	image_ds[(ty+offset)*sharedWidth+(tx+offset)] = image[y*width+x]; //load tile from image

	if (tx < offset) { //right halo pixels loaded
		image_ds[(ty+offset)*sharedWidth + tx + (TILE_WIDTH + offset)] = (haloIndexRight>=width)?0:image[y*width+haloIndexRight];
	}
	if (ty < offset) { //bottom halo pixels from left corner to right corner loaded
		if (tx >= TILE_WIDTH - offset) { //bottom left halo pixels loaded
			image_ds[(ty + (TILE_WIDTH + offset))*sharedWidth + tx - (TILE_WIDTH - offset)] = (haloIndexLeft<0 || haloIndexBottom>=width)?0:image[haloIndexBottom*width+haloIndexLeft]; //bottom left halo pixels for tile loaded
		}
		image_ds[(ty + (TILE_WIDTH + offset))*sharedWidth+tx+offset] = (haloIndexBottom>=width)?0:image[haloIndexBottom*width+x]; //bottom halo cells loaded
		if (tx < offset) {
			image_ds[(ty + (TILE_WIDTH + offset))*sharedWidth + tx + (TILE_WIDTH + offset)] = (haloIndexRight>=width || haloIndexBottom>=width)?0:image[haloIndexBottom*width+haloIndexRight]; //bottom right halo pixels for tile loaded
		}
	}

	__syncthreads();

	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			out += (float)image_ds[(j + ty)*sharedWidth+(i + tx)] * cuda_kernel[j*KERNEL_WIDTH+i];
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_shared_tiled(const unsigned char *image, float *output, size_t width){
	__shared__ float image_ds[(TILE_WIDTH + KERNEL_WIDTH - 1)*(TILE_WIDTH + KERNEL_WIDTH - 1)];
	int sharedWidth = (TILE_WIDTH + KERNEL_WIDTH - 1);
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int x = bx * TILE_WIDTH + tx;
	int y = by * TILE_WIDTH + ty;
	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	//load data
	int haloIndexTop = y - offset;
	int haloIndexBottom = y + offset;
	int haloIndexLeft = x - offset;
	int haloIndexRight = x + offset;

	if (haloIndexTop < 0 || haloIndexLeft < 0) //top left of image
		image_ds[ty*sharedWidth+tx] = 0;
	else // extract top left tile
		image_ds[ty*sharedWidth+tx] = image[y*width+x - offset*width - offset];

	if (haloIndexRight >= width || haloIndexTop < 0) //top right of image
		image_ds[ty*sharedWidth+(tx+offset+offset)] = 0;
	else // extract top right tile
		image_ds[ty*sharedWidth+(tx+offset+offset)] = image[y*width+x - offset*width + offset];

	if (haloIndexBottom >= width || haloIndexLeft < 0) //bottom left of image
		image_ds[(ty+offset+offset)*sharedWidth+tx] = 0;
	else // extract bottom left tile
		image_ds[(ty+offset+offset)*sharedWidth+tx] = image[y*width+x + offset*width - offset];

	if (haloIndexRight >= width || haloIndexBottom >= width) //bottom right of image
		image_ds[(ty+offset+offset)*sharedWidth+(tx+offset+offset)] = 0;
	else // extract bottom right tile
		image_ds[(ty+offset+offset)*sharedWidth+(tx+offset+offset)] = image[y*width+x + offset*width + offset];

	__syncthreads();

	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			out += (float)image_ds[(j + ty)*sharedWidth+(i + tx)] * cuda_kernel[j*KERNEL_WIDTH+i];
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture(float *output, size_t width){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (float i = 0; i < KERNEL_WIDTH; ++i) {
			int mapi = i-offset;
			int mapj = j-offset;
			float u = ((float)x + mapi - (float)width/2)/(float)(width);
			float v = ((float)y + mapj - (float)width/2)/(float)(width);
			out += (float)tex2D(tex2, u+0.5f, v+0.5f) * tex2D(kernelTex2, (float)i/(KERNEL_WIDTH-1), (float)j/(KERNEL_WIDTH-1));
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture_shared(float *output, size_t width){
	__shared__ float image_ds[(TILE_WIDTH + KERNEL_WIDTH - 1)*(TILE_WIDTH + KERNEL_WIDTH - 1)];
	int sharedWidth = (TILE_WIDTH + KERNEL_WIDTH - 1);
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int x = bx * TILE_WIDTH + tx;
	int y = by * TILE_WIDTH + ty;
	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	//top left corner
	image_ds[ty*sharedWidth+tx] = (float)tex2D(tex2, ((float)x - offset - (float)width/2)/(float)(width)+0.5f, ((float)y - offset - (float)width/2)/(float)(width)+0.5f);
	//top right corner
	image_ds[ty*sharedWidth+(tx+offset+offset)] = (float)tex2D(tex2, ((float)x + offset - (float)width/2)/(float)(width)+0.5f, ((float)y - offset - (float)width/2)/(float)(width)+0.5f);
	//bottom left corner
	image_ds[(ty+offset+offset)*sharedWidth+tx] = (float)tex2D(tex2, ((float)x - offset - (float)width/2)/(float)(width)+0.5f, ((float)y + offset - (float)width/2)/(float)(width)+0.5f);
	//bottom right corner
	image_ds[(ty+offset+offset)*sharedWidth+(tx+offset+offset)] = (float)tex2D(tex2, ((float)x + offset - (float)width/2)/(float)(width)+0.5f, ((float)y + offset - (float)width/2)/(float)(width)+0.5f);

	__syncthreads();

	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			out += (float)image_ds[(j + ty)*sharedWidth+(i + tx)] * cuda_kernel[j*KERNEL_WIDTH+i];
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture_1(float *output, size_t width){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			int mapi = i-offset;
			int mapj = j-offset;
			if (mapi + x >= 0 && mapi + x < width && mapj + y >= 0 && mapj + y < width) {
				out += (float)tex1Dfetch(tex, y*width+x+mapj*width+mapi) * tex1Dfetch(kernelTex, j*KERNEL_WIDTH+i);
			}
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture_1_shared(float *output, size_t width){
	__shared__ float image_ds[(TILE_WIDTH + KERNEL_WIDTH - 1)*(TILE_WIDTH + KERNEL_WIDTH - 1)];
	int sharedWidth = (TILE_WIDTH + KERNEL_WIDTH - 1);
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int bx = blockIdx.x;
	int x = bx * TILE_WIDTH + tx;
	int y = by * TILE_WIDTH + ty;
	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	//load data
	int haloIndexTop = y - offset;
	int haloIndexBottom = y + offset;
	int haloIndexLeft = x - offset;
	int haloIndexRight = x + offset;

	if (haloIndexTop < 0 || haloIndexLeft < 0) //top left of image
		image_ds[ty*sharedWidth+tx] = 0;
	else // extract top left tile
		image_ds[ty*sharedWidth+tx] = (float)tex1Dfetch(tex, y*width+x - offset*width - offset);

	if (haloIndexRight >= width || haloIndexTop < 0) //top right of image
		image_ds[ty*sharedWidth+(tx+offset+offset)] = 0;
	else // extract top right tile
		image_ds[ty*sharedWidth+(tx+offset+offset)] = (float)tex1Dfetch(tex, y*width+x - offset*width + offset);

	if (haloIndexBottom >= width || haloIndexLeft < 0) //bottom left of image
		image_ds[(ty+offset+offset)*sharedWidth+tx] = 0;
	else // extract bottom left tile
		image_ds[(ty+offset+offset)*sharedWidth+tx] = (float)tex1Dfetch(tex, y*width+x + offset*width - offset);

	if (haloIndexRight >= width || haloIndexBottom >= width) //bottom right of image
		image_ds[(ty+offset+offset)*sharedWidth+(tx+offset+offset)] = 0;
	else // extract bottom right tile
		image_ds[(ty+offset+offset)*sharedWidth+(tx+offset+offset)] = (float)tex1Dfetch(tex, y*width+x + offset*width + offset);

	__syncthreads();

	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			out += (float)image_ds[(j + ty)*sharedWidth+(i + tx)] * cuda_kernel[j*KERNEL_WIDTH+i];
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture_1_constant(float *output, size_t width){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			int mapi = i-offset;
			int mapj = j-offset;
			if (mapi + x >= 0 && mapi + x < width && mapj + y >= 0 && mapj + y < width) {
				out += (float)tex1Dfetch(tex, y*width+x+mapj*width+mapi) * cuda_kernel[j*KERNEL_WIDTH+i];
			}
		}
	}
	output[y*width+x] = out;
}

__global__ void __launch_bounds__(1024) convolution_texture_constant(float *output, size_t width){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = KERNEL_WIDTH/2; //kernel offset with integer division
	float out = 0;
	for (int j = 0; j < KERNEL_WIDTH; ++j) {
		for (int i = 0; i < KERNEL_WIDTH; ++i) {
			int mapi = i-offset;
			int mapj = j-offset;
			float u = ((float)x + mapi - (float)width/2)/(float)(width);
			float v = ((float)y + mapj - (float)width/2)/(float)(width);
			out += (float)tex2D(tex2, u+0.5f, v+0.5f) * cuda_kernel[j*KERNEL_WIDTH+i];
		}
	}
	output[y*width+x] = out;
}

//verification
int matrixEqual(  float *matrixA, float *matrixB, int n){
  	int bad = 0;
  	for ( int y = 0; y < n && bad < 10 ; y++ )
    	for ( int x = 0; x < n &&  bad < 10 ; x++ ){
			//std::cout << bad <<" " <<  matrixA[y*n+x] << " " << matrixB[y*n+x] << " (" << x << "," << y << ")" << std::endl;
  			if ( abs(matrixA[y*n+x] - matrixB[y*n+x]) > 5e-3 ){
    			bad++;
  			}
		}
  	return !bad;
}

float* GenerateAverageKernel (size_t width) { //generate box averaging filter of width
	float* kernel = (float*)malloc(sizeof(float) * width * width);

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) {
			kernel[i*width+j] = (float)1/(width*width);
		}
	}

	return kernel;
}

float* GenerateEdgeDetectionKernel() {//generate edge detection filter of width 3
	float* kernel = (float*)malloc(sizeof(float) * 3  * 3);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (j == 0) {
				if (i == 1)
					kernel[i*3+j] = -2;
				else
					kernel[i*3+j] = -1;
			} else if (j == 2) {
				if (i == 1)
					kernel[i*3+j] = 2;
				else
					kernel[i*3+j] = 1;
			} else {
				kernel[i*3+j] = 0;
			}
		}
	}
	return kernel;
}

float* GenerateSharpeningKernel() {//generate sharpening filter of width 3
	float* kernel = (float*)malloc(sizeof(float) * 3  * 3);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			if (i == 1 && j == 1)
				kernel[i*3+j] = 9;
			else
				kernel[i*3+j] = -1;
		}
	}
	return kernel;
}

void RescaleIntensity(float *image, unsigned char* output, size_t width) { //rescale pixel intensities into range [0,255]
	float min = image[0];
	float max = image[0];
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) { //get max and min
			if (image[i*width+j] < min) {
				min = image[i*width+j];
			}
			if (image[i*width+j] > max) {
				max = image[i*width+j];
			}
		}
	}
	float contrast = max-min;

	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < width; ++j) { //rescale intensities
			output[i*width+j] = (unsigned char)(255*((image[i*width+j]-min)/contrast));
		}
	}
}

double SequentialConvolution(const char* fileName, const char* outputName, const float* kernel, size_t iterations = 1) {
	unsigned int* width = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* height = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* channels = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned char* data = nullptr;

	__loadPPM(fileName, &data, width, height, channels);

	float* output = (float *) malloc(sizeof(float) * (*width) * (*height));
	unsigned char* outputScaled = (unsigned char*) malloc(sizeof(unsigned char) * (*width) * (*height));

	std::cout << "Image name: " << fileName << std::endl;
	std::cout << "Image size: " << *width << "x" << *height << std::endl;
	imageWidth = *width;

	double average_time = 0;
	for (int i = 0; i < iterations; ++i) {
		memset(output, 0, sizeof(float) * (*width) * (*height));
		struct timespec start, end;
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
			perror( "clock gettime" );
			exit( EXIT_FAILURE );
		}

		convolution_seq(data, kernel, output, *width); //kernel convolution

		if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
		  perror( "clock gettime" );
		  exit( EXIT_FAILURE );
		}
		//compute the time in s
		average_time += (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;
	}
	average_time /= iterations;
	//std::cout << "sequential average time: " << average_time << "s" << std::endl;

	RescaleIntensity(output, outputScaled, *width);

	__savePPM(outputName, outputScaled, *width, *height, *channels);

	free(data);
	free(output);
	free(outputScaled);
	free(width);
	free(height);
	free(channels);
	return average_time; //seconds
}

double GlobalConvolution(const char* fileName, const char* outputName, const float* kernel, size_t iterations = 1) {
	unsigned int* width = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* height = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* channels = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned char* data = nullptr;

	__loadPPM(fileName, &data, width, height, channels);

	const dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
	const dim3 num_blocks((*height) / block_size.x, (*width) / block_size.y);

	cudaMemcpyToSymbol(cuda_kernel, kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH);

	float* output = (float *) malloc(sizeof(float) * (*width) * (*height));
	float* sequential = (float *) malloc(sizeof(float) * (*width) * (*height));
	unsigned char* outputScaled = (unsigned char*) malloc(sizeof(unsigned char) * (*width) * (*height));

	convolution_seq(data, kernel, sequential, *width); //kernel convolution

	float* d_output = 0;
	unsigned char* d_data = 0;
	cudaMalloc((void**)&d_data, sizeof(unsigned char) * (*width) * (*height));
	cudaMalloc((void**)&d_output, sizeof(float) * (*width) * (*height));
	cudaMemcpy(d_data, data, sizeof(unsigned char) * (*width) * (*height), cudaMemcpyHostToDevice);

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	double average_time = 0;
	for (int i = 0; i < iterations; ++i) {
		cudaMemset(d_output, 0, sizeof(float) * (*width) * (*height));

		cudaEventRecord(begin,0);
		convolution_global<<<num_blocks,block_size>>>(d_data, d_output, (*width));
		cudaEventRecord(end,0);
		cudaEventSynchronize(end);
		cudaDeviceSynchronize();
		float time = 0;
		cudaEventElapsedTime(&time, begin, end);
		average_time += time;
	}
	average_time /= iterations;

	cudaMemcpy(output, d_output, sizeof(float)*(*width)*(*height), cudaMemcpyDeviceToHost);

	int equal = matrixEqual(sequential, output, (*width));
	if(!equal) {
		average_time = -1000;
	}

	RescaleIntensity(output, outputScaled, *width);

	__savePPM(outputName, outputScaled, *width, *height, *channels);

	free(data);
	cudaFree(d_data);
	cudaFree(d_output);
	free(output);
	free(outputScaled);
	free(width);
	free(height);
	free(channels);
	return average_time / 1000; //get seconds
}

double SharedConvolution(const char* fileName, const char* outputName, const float* kernel, bool tiled, size_t iterations = 1) {
	unsigned int* width = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* height = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* channels = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned char* data = nullptr;

	__loadPPM(fileName, &data, width, height, channels);

	const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
	const dim3 num_blocks((*height) / block_size.x, (*width) / block_size.y);

	cudaMemcpyToSymbol(cuda_kernel, kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH);

	float* output = (float *) malloc(sizeof(float) * (*width) * (*height));
	float* sequential = (float *) malloc(sizeof(float) * (*width) * (*height));
	unsigned char* outputScaled = (unsigned char*) malloc(sizeof(unsigned char) * (*width) * (*height));

	convolution_seq(data, kernel, sequential, *width); //kernel convolution

	// for (int i = 0; i < 10; ++i) {
	// 	for (int j = 0; j < 10; ++j) {
	// 		std::cout << (int)data[i*(*width) + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	float* d_output = 0;
	unsigned char* d_data = 0;
	cudaMalloc((void**)&d_data, sizeof(unsigned char) * (*width) * (*height));
	cudaMalloc((void**)&d_output, sizeof(float) * (*width) * (*height));
	cudaMemcpy(d_data, data, sizeof(unsigned char) * (*width) * (*height), cudaMemcpyHostToDevice);

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	double average_time = 0;
	for (int i = 0; i < iterations; ++i) {
		cudaMemset(d_output, 0, sizeof(float) * (*width) * (*height));
		if (tiled) {
			cudaEventRecord(begin,0);
			convolution_shared_tiled<<<num_blocks,block_size>>>(d_data, d_output, (*width));
			cudaEventRecord(end,0);
		} else {
			cudaEventRecord(begin,0);
			convolution_shared_strided<<<num_blocks,block_size>>>(d_data, d_output, (*width));
			cudaEventRecord(end,0);
		}
		cudaEventSynchronize(end);

		float time = 0;
		cudaEventElapsedTime(&time, begin, end);
		average_time += time;
	}
	average_time /= iterations;

	cudaMemcpy(output, d_output, sizeof(float)*(*width)*(*height), cudaMemcpyDeviceToHost);

	int equal = matrixEqual(sequential, output, (*width));
	if(!equal) {
		average_time = -1000;
 	}

	RescaleIntensity(output, outputScaled, *width);

	__savePPM(outputName, outputScaled, *width, *height, *channels);

	free(data);
	cudaFree(d_data);
	cudaFree(d_output);
	free(output);
	free(outputScaled);
	free(width);
	free(height);
	free(channels);
	return average_time / 1000; //get seconds
}

double TextureConvolution(const char* fileName, const char* outputName, const float* kernel, bool constant, bool twoD, bool shared, size_t iterations = 1) {
	unsigned int* width = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned int* height = (unsigned int*)malloc(sizeof(unsigned int));
	unsigned char* data = NULL;

	sdkLoadPGM(fileName, &data, width, height);

	const dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
	const dim3 num_blocks((*height) / block_size.x, (*width) / block_size.y);

	const dim3 block_size_tiled(TILE_WIDTH,TILE_WIDTH);
	const dim3 num_blocks_tiled((*height) / block_size_tiled.x, (*width) / block_size_tiled.y);

	cudaMemcpyToSymbol(cuda_kernel, kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH);

	float* output = (float *) malloc(sizeof(float) * (*width) * (*height));
	float* sequential = (float *) malloc(sizeof(float) * (*width) * (*height));
	unsigned char* outputScaled = (unsigned char*) malloc(sizeof(unsigned char) * (*width) * (*height));

	convolution_seq(data, kernel, sequential, *width); //kernel convolution

	float* d_output = 0;
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(float) * (*width) * (*height)));

	unsigned char* d_data = 0;
	cudaMalloc((void**)&d_data, sizeof(unsigned char) * (*width) * (*height));
	cudaMemcpy(d_data, data, sizeof(unsigned char) * (*width) * (*height), cudaMemcpyHostToDevice);

	float* d_kernel = 0;
	checkCudaErrors(cudaMalloc((void**)&d_kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH));
	cudaMemcpy(d_kernel, kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH, cudaMemcpyHostToDevice);
	//texture allocation

	tex.addressMode[0] = cudaAddressModeBorder;
	tex.filterMode = cudaFilterModePoint;

	kernelTex.addressMode[0] = cudaAddressModeClamp;
	kernelTex.filterMode = cudaFilterModePoint;

	checkCudaErrors(cudaBindTexture(NULL, tex, d_data, sizeof(unsigned char) * (*width) * (*height)));
	checkCudaErrors(cudaBindTexture(NULL, kernelTex, d_kernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray *cuImage;
	checkCudaErrors(cudaMallocArray(&cuImage,&channelDesc,*width,*width));
	checkCudaErrors(cudaMemcpyToArray(cuImage,0,0,data,sizeof(unsigned char) * (*width) * (*height),cudaMemcpyHostToDevice));

	tex2.addressMode[0] = cudaAddressModeBorder;
	tex2.addressMode[1] = cudaAddressModeBorder;
	tex2.filterMode = cudaFilterModePoint;
	tex2.normalized = true;

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	float* d_kernel2 = 0;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch((void**)&d_kernel2, &pitch, sizeof(float) * KERNEL_WIDTH, KERNEL_WIDTH));
	checkCudaErrors(cudaMemcpy2D(d_kernel2, pitch, kernel, KERNEL_WIDTH*sizeof(float), KERNEL_WIDTH*sizeof(float), KERNEL_WIDTH,cudaMemcpyHostToDevice) );

	kernelTex2.addressMode[0] = cudaAddressModeClamp;
	kernelTex2.addressMode[1] = cudaAddressModeClamp;
	kernelTex2.filterMode = cudaFilterModePoint;
	kernelTex2.normalized = true;

	checkCudaErrors(cudaBindTextureToArray(tex2, cuImage, channelDesc));
	checkCudaErrors(cudaBindTexture2D(NULL, kernelTex2, d_kernel2, channelDesc2, KERNEL_WIDTH, KERNEL_WIDTH, pitch) );

	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

	double average_time = 0;


	for (int i = 0; i < iterations; ++i) {
		cudaMemset(d_output, 0, sizeof(float) * (*width) * (*height));
		if (!shared) {
			if (!constant) {
				if (twoD) {
					cudaEventRecord(begin,0);
					convolution_texture<<<num_blocks,block_size>>>(d_output, (*width));
					cudaEventRecord(end,0);
				} else {
					cudaEventRecord(begin,0);
					convolution_texture_1<<<num_blocks,block_size>>>(d_output, (*width));
					cudaEventRecord(end,0);
				}
			} else {
				if (twoD) {
					cudaEventRecord(begin,0);
					convolution_texture_constant<<<num_blocks,block_size>>>(d_output, (*width));
					cudaEventRecord(end,0);
				} else {
					cudaEventRecord(begin,0);
					convolution_texture_1_constant<<<num_blocks,block_size>>>(d_output, (*width));
					cudaEventRecord(end,0);
				}
			}
		} else {
			if (twoD) {
				cudaEventRecord(begin,0);
				convolution_texture_shared<<<num_blocks_tiled,block_size_tiled>>>(d_output, (*width));
				cudaEventRecord(end,0);
			} else {
				cudaEventRecord(begin,0);
				convolution_texture_1_shared<<<num_blocks_tiled,block_size_tiled>>>(d_output, (*width));
				cudaEventRecord(end,0);
			}
		}
		cudaEventSynchronize(end);

		float time = 0;
		cudaEventElapsedTime(&time, begin, end);
		average_time += time;
	}
	average_time /= iterations;

	cudaMemcpy(output, d_output, sizeof(float)*(*width)*(*height), cudaMemcpyDeviceToHost);

	int equal = matrixEqual(sequential, output, (*width));
	if(!equal) {
		average_time = -1000;
	}

	RescaleIntensity(output, outputScaled, *width);

	sdkSavePGM(outputName, outputScaled, *width, *height);

	free(data);
	cudaFree(d_output);
	cudaFree(d_data);
	cudaFree(d_kernel);
	cudaFree(d_kernel2);
	cudaFreeArray(cuImage);
	free(output);
	free(outputScaled);
	free(width);
	free(height);
	return average_time / 1000; //get seconds
}

int mainTest(void) {
	float *EdgeKernel = GenerateEdgeDetectionKernel();
	float texture2dTime = TextureConvolution("../data/lena_bw.pgm", "texture_lena_bw_edge.pgm", EdgeKernel, false, true, 10);
	std::cout << "texture2d: " << texture2dTime << "s" << std::endl;
	return 0;
}

int main(void){
	QueryDevice();
	std::cout << std::endl;

	float *AverageKernel = GenerateAverageKernel(KERNEL_WIDTH);
	float *EdgeKernel = GenerateEdgeDetectionKernel();
	float *SharpenKernel = GenerateSharpeningKernel();
	float *kernel;
	if (kernelType[0] == 'a') {
		kernel = AverageKernel;
	} else if (kernelType[0] == 'e') {
		kernel = EdgeKernel;
	} else {
		kernel = SharpenKernel;
	}


	char buffer[100] = "";
	char output[100] = "";
	std::cout << "Kernel size: " << KERNEL_WIDTH << "x" << KERNEL_WIDTH << std::endl;
	std::cout << "Kernel Type: " << kernelType << std::endl;
	strcat(strcat(strcat(buffer, "../data/"), filename),".pgm");
	float sequentialTime = SequentialConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/sequential_"), filename),"_"),kernelType),".pgm"), kernel, 100);
	std::cout << "sequential: " << sequentialTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float globalTime = GlobalConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/global_"), filename),"_"),kernelType),".pgm"), kernel, 100);
	std::cout << "global: " << globalTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float sharedTiledTime = SharedConvolution(buffer, strcat(strcat(strcat(strcat(strcat(strcat(output, "images/shared_"), filename),"_"),kernelType),"_tiled"),".pgm"), kernel, true, 100);
	std::cout << "shared tiled: " << sharedTiledTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float sharedStridedTime = SharedConvolution(buffer, strcat(strcat(strcat(strcat(strcat(strcat(output, "images/shared_"), filename),"_"),kernelType),"_strided"),".pgm"), kernel, false, 100);
	std::cout << "shared strided: " << sharedStridedTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float texture2dTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texture_"), filename),"_"),kernelType),".pgm"), kernel, false, true, false, 100);
	std::cout << "texture2d: " << texture2dTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float texture1dTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texture1_"), filename),"_"),kernelType),".pgm"), kernel, false, false, false, 100);
	std::cout << "texture1d: " << texture1dTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float ConstantTexture2dTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texturec_"), filename),"_"),kernelType),".pgm"), kernel, true, true, false, 100);
	std::cout << "texture2dc: " << ConstantTexture2dTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float ConstantTexture1dTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texture1c_"), filename),"_"),kernelType),".pgm"), kernel, true, false, false, 100);
	std::cout << "texture1dc: " << ConstantTexture1dTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float texture2dSharedTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texture2cs_"), filename),"_"),kernelType),".pgm"), kernel, true, true, true, 100);
	std::cout << "texture2dcshared: " << texture2dSharedTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	float texture1dSharedTime = TextureConvolution(buffer, strcat(strcat(strcat(strcat(strcat(output, "images/texture1cs_"), filename),"_"),kernelType),".pgm"), kernel, true, false, true, 100);
	std::cout << "texture1dcshared: " << texture1dSharedTime << "s" << std::endl; memset(&output[0], 0, sizeof(output));
	std::cout << std::endl;

	float num_ops = imageWidth * imageWidth * KERNEL_WIDTH * KERNEL_WIDTH * 2;
	float sequentialThroughput = num_ops / sequentialTime / 1000000000.0f;
	float globalThroughput = num_ops / globalTime / 1000000000.0f;
	float sharedTiledThroughput = num_ops / sharedTiledTime / 1000000000.0f;
	float sharedStridedThroughput = num_ops / sharedStridedTime / 1000000000.0f;
	float texture2dThroughput = num_ops / texture2dTime / 1000000000.0f;
	float texture1dThroughput = num_ops / texture1dTime / 1000000000.0f;
	float constantTexture2dThroughput = num_ops / ConstantTexture2dTime / 1000000000.0f;
	float constantTexture1dThroughput = num_ops / ConstantTexture1dTime / 1000000000.0f;
	float texture2dSharedThroughput = num_ops / texture2dSharedTime / 1000000000.0f;
	float texture1dSharedThroughput = num_ops / texture1dSharedTime / 1000000000.0f;

	std::cout << "Performance of sequential implementation: " << sequentialThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of global kernel: " << globalThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of shared strided kernel: " << sharedStridedThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of shared tiled kernel: " << sharedTiledThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 2D texture kernel: " << texture2dThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 1D texture kernel: " << texture1dThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 2D texture and constant kernel: " << constantTexture2dThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 1D texture and constant kernel: " << constantTexture1dThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 2D texture shared implementation and constant kernel: " << texture2dSharedThroughput << " GFLOPS" << std::endl;
	std::cout << "Performance of 1D texture shared implementation and constant kernel: " << texture1dSharedThroughput << " GFLOPS" << std::endl;
	std::cout << std::endl;

	// std::cout << "Performance speed-up: global over seqential " << globalThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: strided over seqential " << sharedStridedThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: tiled over seqential " << sharedTiledThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture over seqential " << texture2dThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture over seqential " << texture1dThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over seqential " << constantTexture2dThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over seqential " << constantTexture1dThroughput / sequentialThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: strided over global " << sharedStridedThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: tiled over global " << sharedTiledThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture over global " << texture2dThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture over global " << texture1dThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over global " << constantTexture2dThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over global " << constantTexture1dThroughput / globalThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: tiled over strided " << sharedTiledThroughput / sharedStridedThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture over strided " << texture2dThroughput / sharedStridedThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture over strided " << texture1dThroughput / sharedStridedThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over strided " << constantTexture2dThroughput / sharedStridedThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over strided " << constantTexture1dThroughput / sharedStridedThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture over tiled " << texture2dThroughput / sharedTiledThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture over tiled " << texture1dThroughput / sharedTiledThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over tiled " << constantTexture2dThroughput / sharedTiledThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over tiled " << constantTexture1dThroughput / sharedTiledThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture over 2D texture " << texture1dThroughput / texture2dThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over 2D texture " << constantTexture2dThroughput / texture2dThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over 2D texture " << constantTexture1dThroughput / texture2dThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 2D texture and constant over 1D texture " << constantTexture2dThroughput / texture1dThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over 1D texture " << constantTexture1dThroughput / texture1dThroughput << "x" << std::endl;
	// std::cout << "Performance speed-up: 1D texture and constant over 2D texture and constant " << constantTexture1dThroughput / constantTexture2dThroughput << "x" << std::endl;

	//free(kernel);
	free(AverageKernel);
	free(EdgeKernel);
	free(SharpenKernel);

	return 0;
}
