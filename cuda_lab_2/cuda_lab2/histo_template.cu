// CUDA Runtime
#include <cuda_runtime.h>
#include "../common/helper_cuda.h"
#include "../common/helper_functions.h" 

typedef unsigned int uint;
typedef unsigned char uchar;
#define BIN_NO 256

void histoCPU(uint *h_histo, void *h_data, long byteCount){
    for (uint i = 0; i < BIN_NO; i++)
        h_histo[i] = 0;
    assert(sizeof(uint) == 4 && (byteCount % 4) == 0);
    for (uint i = 0; i < (byteCount >> 2); i++){
        uint data = ((uint *)h_data)[i];
        h_histo[(data >>  0) & 0xFFU]++;
        h_histo[(data >>  8) & 0xFFU]++;
        h_histo[(data >> 16) & 0xFFU]++;
        h_histo[(data >> 24) & 0xFFU]++;
    }
}
/* Kernel using sectioned partitioning */
__global__ void histoGPU_1(uchar *input, uint *d_histo, long size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int section_size = (size - 1) / (blockDim.x * gridDim.x) + 1;
	int start = i * section_size;
	int pos, k;
	for(k = 0; k < section_size; k++){
		if(start + k < size){
			pos = input[start+k]; 
			if(pos >=0 && pos < 256)
				atomicAdd(&(d_histo[pos]), 1);
		}
	}
}

/* Complete the following kernel using the interleaved partitioning. */
//__global__ void histoGPU_2(uchar *input, uint *d_histo, long size){
//
//}

/* Complete the following kernel using shared memory */
//__global__ void histoGPU_3(uchar *input, uint *d_histo, long size, uint num_bins){
//	
//}

int main(int argc, char **argv)
{
    uchar *h_data;
    uint  *h_histoCPU, *h_histoGPU;
    uchar *d_data;
    uint  *d_histo;
    StopWatchInterface *hTimer = NULL;
    int PassFailFlag = 1;
    long byteCount = 1 << 24;
    uint uiSizeMult = 1;
	int numRuns = 2;
	const static char *funcName;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    // set logfile name and start logs
    printf("[%s] - Starting...\n", "histogram");

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
           deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
    sdkCreateTimer(&hTimer);
    // Optional Command-line multiplier to increase size of array to histogram
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")){
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1,MIN(uiSizeMult, 10));
        byteCount *= uiSizeMult;
    }
    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_data         = (uchar *)malloc(byteCount);
    h_histoCPU = (uint *)malloc(BIN_NO * sizeof(uint));
    h_histoGPU = (uint *)malloc(BIN_NO * sizeof(uint));
    printf("...generating input data\n");
    srand(2019);
    for (uint i = 0; i < byteCount; i++){
    	h_data[i] = rand() % 256;
    }
	for(uint i = 0; i < BIN_NO; i++){
		h_histoGPU[i] = 0;
	}
	dim3 blk_size(512);
	//dim3 grid_size((BIN_NO + blk_size.x - 1)/blk_size.x);
	dim3 grid_size(32);
    printf("...allocating GPU memory and copying input data\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_histo, BIN_NO * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, byteCount, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(d_histo, h_histoGPU, BIN_NO * sizeof(uint), cudaMemcpyHostToDevice));
    printf("Starting up 256-bin histogram...\n\n");       
	printf("Running 256-bin GPU histogram for %ld bytes (%d runs)...\n\n", byteCount, numRuns);
	funcName = "[histoGPU_3]\0";
    for (int iter = 0; iter < numRuns; iter++){
		//iter == -1 -- warmup iteration
		if (iter == 0){
			cudaDeviceSynchronize();
			sdkResetTimer(&hTimer);
			sdkStartTimer(&hTimer);
		}
		checkCudaErrors(cudaMemset(d_histo, 0, BIN_NO * sizeof(uint)));
		//checkCudaErrors(cudaMemcpy(d_histo, h_histoGPU, BIN_NO * sizeof(uint), cudaMemcpyHostToDevice));
		histoGPU_1<<<grid_size, blk_size, BIN_NO * sizeof(uint)>>>(d_data, d_histo, byteCount);
		//histoGPU_2<<<grid_size, blk_size>>>(d_data, d_histo, byteCount);
		//histogram_kernel<<<grid_size, blk_size, BIN_NO * sizeof(uint)>>>(d_data, d_histo, byteCount, BIN_NO);
	}
	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer);
	double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
	printf("%s time (average) : %.5f sec, %.4f MB/sec\n\n",funcName, dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
	printf("%s, Throughput = %.4f MB/s, Time = %.5f s, Size = %ld Bytes, NumDevsUsed = %u, Workgroup = %u\n",funcName, (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, 512);
	printf("\nValidating GPU results...\n");
	printf(" ...reading back GPU results\n");
	checkCudaErrors(cudaMemcpy(h_histoGPU, d_histo, BIN_NO * sizeof(uint), cudaMemcpyDeviceToHost));
	printf(" ...histoCPU()\n");
	histoCPU(h_histoCPU, h_data, byteCount);
	printf(" ...comparing the results...\n");
	for (uint i = 0; i < 256; i++){
		if (h_histoGPU[i] != h_histoCPU[i]){
			PassFailFlag = 0;
			printf("%u, %u --", h_histoGPU[i], h_histoCPU[i]);
		}
	}
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
	printf("Cleaning up...\n");
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_histo));
    checkCudaErrors(cudaFree(d_data));
    free(h_histoGPU);
    free(h_histoCPU);
    free(h_data);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    printf("%s - Test Summary\n", "histogram");

    // pass or fail 
    if (!PassFailFlag){
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
