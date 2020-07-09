
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

#define SAMPLES 242
#define NO_NODES_IN_INPUT 14
#define NO_NODES_IN_DEST 13
#define dim max(NO_NODES_IN_INPUT, NO_NODES_IN_DEST)
#define ALPHA 0.5
//nvcc -arch=sm_35 -rdc=true kernel.cu -o proj -lcudadevrt


__device__ int NO_LAYERS;
__device__ int EPOCHS;
//layer specifies what column in activation matrix is the input!
__device__ void MatVectMultiplication(float* device_Mat, float* activation_matrix, int layer)
{
	

    __shared__ float cache[1024];
    int tindex = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < (NO_NODES_IN_INPUT + blockDim.x - 1) / blockDim.x; i++) {
        cache[threadIdx.x] = *(activation_matrix + (i * blockDim.x + threadIdx.x) * NO_LAYERS + layer);
        __syncthreads();
        for (int i = 0; i < NO_NODES_IN_INPUT; i++) {
            if (tindex < NO_NODES_IN_DEST) {
                activation_matrix[(tindex + 1) * NO_LAYERS + (layer + 1)] += cache[i] * device_Mat[i * dim + tindex];
            }
        }
    }
    activation_matrix[(tindex + 1) * NO_LAYERS + (layer + 1)] = (float)1 / (1 + exp(-1 * activation_matrix[(tindex + 1) * NO_LAYERS + (layer + 1)]));
    activation_matrix[0 * NO_LAYERS + (layer + 1)] = 1;
}

__device__ void BackProp(float* weight_tensor, float* activation_matrix, int layer)
{

    int tindex = blockIdx.x * blockDim.x + threadIdx.x;

    float my_old_activation = *(activation_matrix + tindex * NO_LAYERS + (layer - 1));
    //This sorts out the weight update
    for (int i = 0; i < NO_NODES_IN_DEST; i++) {
        *(weight_tensor + (0) * dim * dim + tindex * dim + i) = *(weight_tensor + layer * dim * dim + tindex * dim + i);
        *(weight_tensor + layer * dim * dim + tindex * dim + i) = *(weight_tensor + layer * dim * dim + tindex * dim + i) - ALPHA * *(activation_matrix + (i + 1) * NO_LAYERS + layer) * my_old_activation;
    }

    //We then go on to do the activation update
    //First do the updated activation. The old activation is saved. This is waht will go in the function
    float sum = 0;
    for (int i = 0; i < NO_NODES_IN_DEST; i++) {
        // Fix this 2           // This 1 is in relation to the layer we are in.
        sum += *(activation_matrix + (i + 1) * NO_LAYERS + layer) * *(weight_tensor + 0 * dim * dim + tindex * dim + i);
    }
    *(activation_matrix + tindex * NO_LAYERS + (layer - 1)) = sum * my_old_activation * (1 - my_old_activation);
}

__device__ void separate_reduction(float* weight_matrix, float* activation_matrix)
{

    __shared__ float share[1024];

    int tindex = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    share[lid] = 0;

    if (tindex < NO_NODES_IN_INPUT) {
        share[lid] = *(weight_matrix + tindex * dim + 0) * *(activation_matrix + tindex * NO_LAYERS + (NO_LAYERS - 2));
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lid < s) {
            share[lid] += share[lid + s];
        }
        __syncthreads();
    }
    if (lid == 0) {
        *(activation_matrix + blockIdx.x * NO_LAYERS + (NO_LAYERS - 1)) = share[0];
    }
}

__global__ void NeuralNetwork(float* feature_vector, float* weight_tensor, float* activation_matrix, float target)
{

    int tindex = blockIdx.x * blockDim.x + threadIdx.x;

    if (tindex < NO_NODES_IN_INPUT) {
        *(activation_matrix + tindex * NO_LAYERS + 0) = *(feature_vector + tindex);
    }
    __syncthreads();

    //Forward Prop (100%)
    for (int i = 1; i < NO_LAYERS - 1; i++) {
        MatVectMultiplication((weight_tensor + i * dim * dim), activation_matrix, i - 1);
    }

    separate_reduction((weight_tensor + (NO_LAYERS - 1) * dim * dim), activation_matrix);

    //The serial part to calculate the activation of the last layer and to calculate the error
    if (tindex == 0) {
        float sum = 0;
        for (int i = 0; i < NO_NODES_IN_INPUT; i++) {
            sum +=  *(activation_matrix + i * NO_LAYERS + (NO_LAYERS - 1));
        }
		
        *(activation_matrix + 0 + NO_LAYERS - 1) = sum;
        *(activation_matrix + 0 * NO_LAYERS + NO_LAYERS - 1) = (float)1 / (1 + exp(-1 * *(activation_matrix + 0 + NO_LAYERS - 1)));
        *(activation_matrix + 1 * NO_LAYERS + NO_LAYERS - 1) = *(activation_matrix + 0 * NO_LAYERS + NO_LAYERS - 1) - target;
        *(activation_matrix + 2 * NO_LAYERS + NO_LAYERS - 1) = *(activation_matrix + 1 * NO_LAYERS + NO_LAYERS - 1) ;
    }

    //This is just for the first layer from the back because we are only pulling a compount activation form one source in this layer!
    if (tindex < NO_NODES_IN_INPUT) {
        *(weight_tensor + 0 * dim * dim + tindex * dim + 0) = *(weight_tensor + (NO_LAYERS - 1) * dim * dim + tindex * dim + 0); //Correct for all networks!                                                  //this 2 belongs here( this is where the activation of the output is stored for all networks!)
        *(weight_tensor + (NO_LAYERS - 1) * dim * dim + tindex * dim + 0) = *(weight_tensor + (NO_LAYERS - 1) * dim * dim + tindex * dim + 0) - ALPHA * (*(activation_matrix + 2 * NO_LAYERS + (NO_LAYERS - 1)) * *(activation_matrix + tindex * NO_LAYERS + (NO_LAYERS - 2)));
		
        float my_old_activation = *(activation_matrix + tindex * NO_LAYERS + (NO_LAYERS - 2)); //one layer back from the end( is related to
        //this 1
        *(activation_matrix + tindex * NO_LAYERS + (NO_LAYERS - 2)) = *(activation_matrix + 2 * NO_LAYERS + (NO_LAYERS - 1)) * *(weight_tensor + 0 + tindex * dim + 0) * my_old_activation * (1 - my_old_activation);
		
        for (int i = NO_LAYERS - 2; i >=1; i--) {
            BackProp(weight_tensor, activation_matrix, i);
        }
		
    }
	
}

__global__ void entry_NN(float* feature_vectors, float* weight_tensor, float* activation_matrix, float* target_vector)
{

    int tindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (tindex == 0) {
        dim3 blocks((dim + 1024 - 1) / 1024);
        dim3 threads(1024);
		for(int j=0;j<EPOCHS;j++){
        for (int i = 0; i < SAMPLES; i++) {
            NeuralNetwork<<<blocks, threads>>>((feature_vectors), weight_tensor, activation_matrix, *(target_vector + i));
            cudaDeviceSynchronize();
        }
		}
    }
}


int calculations(int layers);
float sigmoid(float x);
float activation(float* theta, float* feature, int size);
float activationDifferential(float value);
void update(float* weight, float p, float error, float partial);
float neuralNetSerial(float* feature, float* theta, int layers, float bias);
float learnSerial(float* feature, float* theta, int layers, float bias, int max_epoch, float epsilon, float* target, int size);



int main(int argc, char ** argv)
{

	int layers=50;
	int epochs=1;
	
	printf("Please enter the number of layers in the network (excluding the input layer):");
	scanf("%d",&layers);

	printf("Please enter the number of epochs:");
	scanf("%d",&epochs);
	
	layers=layers+1;
	
	cudaMemcpyToSymbol(NO_LAYERS,&layers,sizeof(int),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(EPOCHS,&epochs,sizeof(int),0,cudaMemcpyHostToDevice);



	
	
    //Import the data
    float* feature = (float*)malloc(sizeof(float) * SAMPLES * NO_NODES_IN_INPUT); //242 values
    float* target = (float*)malloc(sizeof(float) * SAMPLES);

		
    FILE* fp = fopen("training.csv", "r");
    char buf[1024];

    int row_count = -2;

    while (fgets(buf, 1024, fp)) {
        row_count++;
        if (row_count == -1)
            continue;
        int columnCount = 0;
        char* field = strtok(buf, ",");
        while (field) {
            if (columnCount == 0) {
                feature[row_count * NO_NODES_IN_INPUT + columnCount] = 1; //bias
            }
            else if (columnCount < NO_NODES_IN_INPUT && columnCount > 0) {
                feature[row_count * NO_NODES_IN_INPUT + columnCount] = atof(field);
            }
            else if (columnCount == NO_NODES_IN_INPUT) {
                target[row_count] = atof(field);
            }

            field = strtok(NULL, ",");
            columnCount++;
        }
    }

    fclose(fp);
	

    int nodes_in_matrix = dim * dim;
    float* weight_tensor = (float*)malloc(layers * nodes_in_matrix * sizeof(float));
	
	
	
    srand(time(NULL));
    // for a way to programmatically produces theta values
    for (int i = 1; i <= layers - 1; i++) {
        for (int j = 0; j < NO_NODES_IN_INPUT; j++) {
            for (int k = 0; k < NO_NODES_IN_DEST; k++) {
                *(weight_tensor + i * dim * dim + j * dim + k) = (float)rand() / (RAND_MAX);
            }
        }
    }

	
    float *d_features, *d_weight_matrix;
    cudaMalloc((void**)&d_features, sizeof(float) * SAMPLES * NO_NODES_IN_INPUT);
    cudaMemcpy(d_features, feature,sizeof(float) * SAMPLES * NO_NODES_IN_INPUT, cudaMemcpyHostToDevice);
	

    cudaMalloc((void**)&d_weight_matrix, layers * nodes_in_matrix * sizeof(float));
    cudaMemcpy(d_weight_matrix, weight_tensor, layers * nodes_in_matrix * sizeof(float), cudaMemcpyHostToDevice); 

    float *h_activation_matrix, *d_activation_matrix;
    h_activation_matrix = (float*)malloc(NO_NODES_IN_INPUT * layers * sizeof(float));
    memset(h_activation_matrix, 0, NO_NODES_IN_INPUT * layers * sizeof(float));

    cudaMalloc((void**)&d_activation_matrix, NO_NODES_IN_INPUT * layers * sizeof(float));
    cudaMemcpy(d_activation_matrix, h_activation_matrix, NO_NODES_IN_INPUT * layers * sizeof(float), cudaMemcpyHostToDevice);
	h_activation_matrix = (float*)malloc(NO_NODES_IN_INPUT * layers * sizeof(float));
	
	
	float* d_targets;
	cudaMalloc((void**)&d_targets,SAMPLES*sizeof(float));
	cudaMemcpy(d_targets,target,SAMPLES*sizeof(float),cudaMemcpyHostToDevice);
	
	

    dim3 blocks((dim + 1024 - 1) / 1024);
    dim3 threads(1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    entry_NN<<<blocks, threads>>>(d_features, d_weight_matrix, d_activation_matrix, d_targets);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float parallel_time = 0;
    cudaEventElapsedTime(&parallel_time, start, stop);
	parallel_time=(float) parallel_time/1000;
	

    float* thetaOriginal =(float*) malloc(sizeof(float) * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT * layers);
    float* theta = (float* )malloc(sizeof(float) * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT * layers);


    srand(time(NULL));
        for (int l = 0; l < layers; l++) {
            for (int j = 0; j < NO_NODES_IN_INPUT; j++) {
                for (int i = 1; i <= NO_NODES_IN_INPUT; i++) { //initialize weights randomly for stochastic descent
                    thetaOriginal[l * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT + j * NO_NODES_IN_INPUT + (i - 1)] = (float)rand() / (float)RAND_MAX * 2 - 1;
                    //printf("%d,%f\n", l*SIZE*SIZE + j*SIZE + (i-1), thetaOriginal[l*SIZE*SIZE + j*SIZE + (i-1)]);
                }
            }
		}

	    float bias = 1;
	    const float epsilon = pow(10, -7);
		


        memcpy(theta, thetaOriginal, sizeof(float) * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT * layers);
        double begin = clock();
        float ans =  learnSerial(feature,theta,layers,bias,epochs,epsilon, target,  SAMPLES);
        double end = clock();
	
		double serial_time=(double)(end-begin)/CLOCKS_PER_SEC;
	
		int calcs=calculations(layers)*epochs*SAMPLES;
	
		printf("Serial Time(in seconds):%fs\n",serial_time);
		printf("Parallel Time(in seconds):%fs\n",parallel_time);
		
		printf("Speedup: %fx\n",(float) serial_time/parallel_time);
		printf("Serial Throughput: %f GFLOPS/s\n",(float)calcs/(serial_time*1000000000));
		printf("Parallel Throughput: %f GFLOPS/s\n",(float)calcs/(parallel_time*1000000000));
	
	
	
	

	//To print the weight_matrix
	/*
    for (int i = 1; i < layers; i++) {
        for (int j = 0; j < NO_NODES_IN_INPUT; j++) {
            for (int k = 0; k < NO_NODES_IN_DEST; k++) {
                printf("%f ", *(weight_tensor + i * (nodes_in_matrix) + j * dim + k));
            }
            printf("\n");
        }
        printf("\n\n");
    }

	
    //To test the activation matrix
    printf("\n\n");
    cudaMemcpy(h_activation_matrix, d_activation_matrix, NO_NODES_IN_INPUT * layers * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NO_NODES_IN_INPUT; i++) {
        for (int j = 0; j < layers; j++) {
            printf("%f ", *(h_activation_matrix + i * layers + j));
        }
        printf("\n");
    }
	*/
	

	
	free(feature);
	free(target);
	free(weight_tensor);
	cudaFree(d_features);
	cudaFree(d_weight_matrix);
	free(h_activation_matrix);
	cudaFree(d_activation_matrix);
	cudaFree(d_targets);


    return 0;
}


int calculations(int layers) {
    int i = NO_NODES_IN_INPUT;
    int forward = ((i-1)*(i+(i-1)+3))*(layers-1) + (i+(i-1)+3) + i*4 + i*(i-1)*8;
    int back = (((i-1)*6+6)*(i*(i-1)))*(layers-2);
    return forward+back;
}
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float activation(float* theta, float* feature, int size) {
    float x = 0;
    for (int i = 0; i < size; i++) {
        x += theta[i * size] * feature[i];
    }
    return sigmoid(x);
}

float activationDifferential(float value) { //differential of activation function
    return value * (1 - value);
}

float error(float target, float actual) {
    return actual - target; //differential for mean squared error
    //return pow(actual-target,2)/2.0; //mean squared error
}

void update(float* weight, float p, float error, float partial) {
    *weight = *weight - ALPHA * error * partial * p; //mean squared error
}

float neuralNetSerial(float* feature, float* theta, int layers, float bias) {
    float a[layers - 1][NO_NODES_IN_INPUT];
    a[0][0] = bias;
    for (int i = 1; i < NO_NODES_IN_INPUT; i++) {
        a[0][i] = activation(&theta[i], feature, NO_NODES_IN_INPUT);
        //printf("%d, %f\n", i, a[0][i]);
    }
    int finalLayer = layers - 1;
    if (layers > 2) {
        for (int l = 1; l < finalLayer; l++) {
            a[l][0] = bias;
            for (int i = 1; i < NO_NODES_IN_INPUT; i++) {
                a[l][i] = activation(&theta[i + l * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT], a[l - 1], NO_NODES_IN_INPUT);
            }
        }
    }
    float out = 0;
    for (int i = 0; i < NO_NODES_IN_INPUT; i++) {
        out += a[finalLayer - 1][i] * theta[finalLayer * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT + i * NO_NODES_IN_INPUT];
        //printf("%d, %f,%f,%f\n", i, a[finalLayer-1][i] * theta[finalLayer*NO_NODES_IN_INPUT*NO_NODES_IN_INPUT + i*NO_NODES_IN_INPUT],theta[finalLayer*NO_NODES_IN_INPUT*NO_NODES_IN_INPUT + i*NO_NODES_IN_INPUT],a[finalLayer-1][i]);
    }
    //printf("%f\n", out);
    float final = sigmoid(out); //forward propogate
    return final;
}

float learnSerial(float* feature, float* theta, int layers, float bias, int max_epoch, float epsilon, float* target, int size) {
    int epoch = 1;
    float final = 0;

    while (1) {
        for (int value = 0; value < size; value++) {
            //forward
            float a[layers - 1][NO_NODES_IN_INPUT];
            a[0][0] = bias;
            for (int i = 1; i < NO_NODES_IN_INPUT; i++) {
                a[0][i] = activation(&theta[i], &feature[value * NO_NODES_IN_INPUT], NO_NODES_IN_INPUT);
            }
            int finalLayer = layers - 1;
            if (layers > 2) {
                for (int l = 1; l < finalLayer; l++) {
                    a[l][0] = bias;
                    for (int i = 1; i < NO_NODES_IN_INPUT; i++) {
                        a[l][i] = activation(&theta[i + l * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT], a[l - 1], NO_NODES_IN_INPUT);
                    }
                }
            }
            float out = a[finalLayer - 1][0] * theta[finalLayer * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT];
            for (int i = 1; i < NO_NODES_IN_INPUT; i++) {
                out += a[finalLayer - 1][i] * theta[finalLayer * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT + i * NO_NODES_IN_INPUT];
                //printf("%d, %f\n", i, a[finalLayer-1][i]);
            }
            final = sigmoid(out); //forward propogate
            //if (final == previous) break; //test convergence

            if (fabs(final - target[value]) < epsilon) break; //below error requirement
            //===========================================================
            //back propogation

            //learn
            int layer = finalLayer - 1;
            float e = error(target[value], final);
            float back[layers - 1][NO_NODES_IN_INPUT - 1][NO_NODES_IN_INPUT - 1];
            for (int i = layers * NO_NODES_IN_INPUT * NO_NODES_IN_INPUT - 1; i >= 0; i--) {
                if (layer != -1) {
                    if (layer == finalLayer - 1) { //last layer (can be 0(2layers) or 1(3layers))
                        if (i % NO_NODES_IN_INPUT == 0) { //works out the thetas corresponding to the final node
                            if (i % (NO_NODES_IN_INPUT * NO_NODES_IN_INPUT) > 0) { //thetas corresponding to the non-bias nodes
                                back[layer][(i / NO_NODES_IN_INPUT) % NO_NODES_IN_INPUT - 1][0] = activationDifferential(final) * theta[i];
                            }
                            update(&theta[i], a[layer][i % (NO_NODES_IN_INPUT * NO_NODES_IN_INPUT) / NO_NODES_IN_INPUT], e, activationDifferential(final));//update theta first then add to back
                            //printf("%d, %f, %f, %f, %f\n", i ,theta[i], a[layer][i%(NO_NODES_IN_INPUT*NO_NODES_IN_INPUT)/NO_NODES_IN_INPUT], e,activationDifferential(final));
                            //printf("%f\n", activationDifferential(a[layer][i%(NO_NODES_IN_INPUT*NO_NODES_IN_INPUT)/NO_NODES_IN_INPUT]));
                        }
                    }
                    else { //layer 0 to finalLayer-1
                        if (i % NO_NODES_IN_INPUT > 0) { //ignores the bias node weights for next layer
                            float b = 0;
                            if (layer + 1 == finalLayer - 1) { //final layer was last layer
                                b = back[layer + 1][i % NO_NODES_IN_INPUT - 1][0];
                            }
                            else { //add all differentials
                                for (int w = 0; w < NO_NODES_IN_INPUT - 1; w++) {
                                    b += back[layer + 1][i % NO_NODES_IN_INPUT - 1][w];
                                }
                            }
                            float last = activationDifferential(a[layer + 1][i % NO_NODES_IN_INPUT]); //last z differential;
                            back[layer][(int)(i % (NO_NODES_IN_INPUT * NO_NODES_IN_INPUT) / NO_NODES_IN_INPUT) - 1][i % NO_NODES_IN_INPUT - 1] = last * theta[i] * b; //update next layer differentials
                            update(&theta[i], a[layer][i % (NO_NODES_IN_INPUT * NO_NODES_IN_INPUT) / NO_NODES_IN_INPUT], e, last * b);
                            //printf("S%d, %f, %f, %f, %f, %f, %f\n", i,theta[i], a[layer][i%(NO_NODES_IN_INPUT*NO_NODES_IN_INPUT)/NO_NODES_IN_INPUT], e,last, b, last*b);
                        }
                    }
                }
                else { //first layer (-1)
                    if (i % NO_NODES_IN_INPUT > 0) { //ignores the bias node weights for next layer
                        float b = 0;
                        if (layer + 1 == finalLayer - 1) { //final layer was last layer
                            b = back[layer + 1][i % NO_NODES_IN_INPUT - 1][0];
                        }
                        else { //add all differentials
                            for (int w = 0; w < NO_NODES_IN_INPUT - 1; w++) {
                                b += back[layer + 1][i % NO_NODES_IN_INPUT - 1][w];
                            }
                        }
                        float last = activationDifferential(a[layer + 1][i % NO_NODES_IN_INPUT]); //last z differential;
                        update(&theta[i], feature[value * NO_NODES_IN_INPUT + (int)(i / NO_NODES_IN_INPUT)], e, last * b); //update on features
                        //printf("S%d, %f, %f, %f, %f, %f\n", i, theta[i], feature[(int)(i/NO_NODES_IN_INPUT)], e, last, b);
                    }
                }
                if (i % (NO_NODES_IN_INPUT * NO_NODES_IN_INPUT) == 0) { //last theta of each layer
                    layer--; //move back a layer
                }
            }
        }
        epoch++;
        if (epoch >= max_epoch) break; //max epoch reached
    }
    return neuralNetSerial(feature, theta, layers, bias);
}


int prediction(float value) {
    if (value >= 0.5) return 1;
    else return 0;
}

float serialAccuracy(float* feature, float* theta, int layers, float bias, float* target, int size) {
    int accurateSample = 0;
    for (int i = 0; i < size; i++) {
        float output = neuralNetSerial(&feature[i * NO_NODES_IN_INPUT], theta, layers, bias);
        if (prediction(output) == target[i])
            accurateSample++;
    }
    return (float)accurateSample / (float)size;
}


