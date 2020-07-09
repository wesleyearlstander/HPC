#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define ALPHA 10
#define SIZE 3


float sigmoid(float x)
{
     return 1 / (1 + exp((double) -x));
}

float activation (float* theta, float* feature, int size) {
    float x = 0;
    for (int i = 0; i < size; i++) {
        x+=theta[i*size]*feature[i];
    }
    return sigmoid(x);
}

float activationDifferential (float value) { //differential of activation function
    return value*(1-value);
}

float error (float target, float actual) {
    return actual - target;
    //return pow(actual-target,2)/2.0; //mean squared error
}

void update(float* weight, float p, float error, float partial) {
    *weight = *weight - ALPHA*error*partial*p; //mean squared error
}

float serial(float* feature, float* theta, int layers) {
    float a[layers-1][SIZE];
    a[0][0] = 1;
    for (int i = 1; i < SIZE; i++) {
        a[0][i] = activation(&theta[i], feature, SIZE);
    }
    int finalLayer = layers-1;
    if (layers>2) {
        for (int l = 1; l < finalLayer; l++) {
            a[l][0] = 1;
            for (int i = 1; i < SIZE; i++) {
                a[l][i] = activation(&theta[i + l*SIZE*SIZE], a[l-1], SIZE);
            }
        }
    }
    float out = a[finalLayer-1][0] * theta[finalLayer*SIZE*SIZE];
    for (int i = 1; i < SIZE; i++) {
        out += a[finalLayer-1][i] * theta[finalLayer*SIZE*SIZE + i*SIZE];
    }
    float final = sigmoid(out); //forward propogate
    return final;
}



int main(int argc, char *argv[]){
    int world_size, world_rank;
	int sendcount, recvcount, source;
    int layers = 2;
	float recvbufT[layers][SIZE];
    float recvbufF;
    float target = 0.8;

    float* feature = malloc(sizeof(float)*SIZE);
    float* thetaOriginal = malloc(sizeof(float)*SIZE*SIZE*layers);
    float* theta = malloc(sizeof(float)*SIZE*SIZE*layers);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank ==0) {
        if (world_size != SIZE) {
            printf("Make processes = %d\n", SIZE);
            return-1;
        }
    }

    if(world_rank==0){ //serial part
        for (int l = 0; l < layers; l++) {
            for (int j = 0; j < SIZE; j++) {
                if (l == 0)
                    feature[j] = j+1; //start on 1 for bias
                for (int i = 1; i <= SIZE; i++) {
                    thetaOriginal[l*SIZE*SIZE + j*SIZE + (i-1)] = 1/(float)((j*SIZE) + i);
                }
            }
        }
        memcpy(theta, thetaOriginal, sizeof(float)*SIZE*SIZE*layers);

        printf("SerialPRE:%f\n", serial(feature,theta,layers)); //output before training
    }



    if (world_rank==0) { //learning
        int epoch = 1;
        int max_epoch = 2;
        float final = 0;
        float previous = 0;
        while (1) {
            //forward
            float a[layers-1][SIZE];
            a[0][0] = 1;
            for (int i = 1; i < SIZE; i++) {
                a[0][i] = activation(&theta[i], feature, SIZE);
            }
            int finalLayer = layers-1;
            if (layers>2) {
                for (int l = 1; l < finalLayer; l++) {
                    a[l][0] = 1;
                    for (int i = 1; i < SIZE; i++) {
                        a[l][i] = activation(&theta[i + l*SIZE*SIZE], a[l-1], SIZE);
                    }
                }
            }
            float out = a[finalLayer-1][0] * theta[finalLayer*SIZE*SIZE];
            for (int i = 1; i < SIZE; i++) {
                out += a[finalLayer-1][i] * theta[finalLayer*SIZE*SIZE + i*SIZE];
            }
            final = sigmoid(out); //forward propogate
            if (final == previous) break; //test convergence
            previous = final;
            if(fabs(final - target)<pow(10,-5)) break; //below error requirement
            if(epoch >= max_epoch) break; //max epoch reached



            //===========================================================
            //back propogation

            //learn
            int layer = finalLayer-1;
            float e = error(target, final);
            float back[layers-1][SIZE-1][SIZE-1];
            for (int i = layers*SIZE*SIZE-1; i>=0;i--) {
                if (layer != -1) {
                    if (layer == finalLayer-1) { //last layer (can be 0(2layers) or 1(3layers))
                        if (i%SIZE==0) { //works out the back values for non bias nodes
                            update(&theta[i], a[layer][i%(SIZE*SIZE)/SIZE], e, activationDifferential(final));//update theta first then add to back
                            if (i%(SIZE*SIZE)>0) { //thetas corresponding to the non-bias nodes
                                back[layer][(i/SIZE)%SIZE-1][0] = activationDifferential(final)*theta[i];
                                printf("%d,%f,%f,%f\n",i,back[layer][(i/SIZE)%SIZE-1][0], activationDifferential(final), theta[i]);
                            }
                        }
                    }
                    else { //layer 0 to finalLayer-1
                        if (i%SIZE>0) { //ignores the bias node weights
                            float b = 0;
                            if (layer+1 == finalLayer-1) { //final layer was last layer
                                b = back[layer+1][i%SIZE-1][0];
                            } else { //add all differentials
                                for (int w = 0; w < SIZE-1; w++) {
                                    b += back[layer+1][i%SIZE-1][w];
                                }
                            }
                            float last = activationDifferential(a[layer+1][i%SIZE]); //last z differential;
                            update(&theta[i], a[layer][i%SIZE], e, last*b);
                            back[layer][(int)(i%(SIZE*SIZE)/SIZE)-1][i%SIZE-1] = last*theta[i]*b; //update next layer differentials
                        }
                    }
                }
                else { //first layer (-1)
                    if (i%SIZE>0) { //ignores the bias node weights
                        float b = 0;
                        if (layer+1 == finalLayer-1) { //final layer was last layer
                            b = back[layer+1][i%SIZE-1][0];
                        } else { //add all differentials
                            for (int w = 0; w < SIZE-1; w++) {
                                b += back[layer+1][i%SIZE-1][w];
                            }
                        }
                        float last = activationDifferential(a[layer+1][i%SIZE]); //last z differential;
                        update(&theta[i], feature[(int)(i/SIZE)], e, last*b); //update on features
                        printf("%d,%f,%f,b%f,f%f\n", i,b, theta[i],b,feature[(int)(i/SIZE)]);
                    }
                }
                //printf("[%d,%f],", i,theta[i]);
                if (i%(SIZE*SIZE)==0) {
                    layer--; //move back a layer
                }
            }
            epoch++;
            //printf("%d,%f\n", epoch, e);
        }
        printf("Serial:%f\n", final);

        memcpy(theta, thetaOriginal, sizeof(float)*SIZE*SIZE*layers);
    }

    //distribute theta
    for (int i = 0; i < layers; i++) {
        MPI_Scatter(&theta[i*SIZE*SIZE], SIZE, MPI_FLOAT, &recvbufT[i][0], SIZE, MPI_FLOAT,0,MPI_COMM_WORLD);
    }

    float pv[SIZE];
    float v[SIZE];
    float result[layers-1][SIZE];
    float finalResult;

    int layerCount = 0;
    //distribute feature vector
    MPI_Scatter(feature, 1, MPI_FLOAT, &recvbufF, 1, MPI_FLOAT,0,MPI_COMM_WORLD);
    if (world_rank == 0) recvbufF = 1; //bias
    for (int i = 0; i < SIZE; i++) { //first set of activation functions
        pv[i] = recvbufT[layerCount][i] * recvbufF;
    }
    layerCount++;

    MPI_Barrier(MPI_COMM_WORLD);
    if (layers>2){ //middle layers
        for (int i = 1; i < layers-1; i++) {
            MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT,MPI_COMM_WORLD);
            result[layerCount-1][world_rank] = 0;
            if (world_rank != 0) {
                for (int i = 0; i < SIZE; i++) {
                    result[layerCount-1][world_rank] += v[i];
                }
                result[layerCount-1][world_rank] = sigmoid(result[layerCount-1][world_rank]);
            } else {
                result[layerCount-1][world_rank] = 1;
            }
            for (int i = 0; i < SIZE; i++) {
                pv[i] = recvbufT[layerCount][i] * result[layerCount-1][world_rank];
            }
            layerCount++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT, MPI_COMM_WORLD);
    result[layerCount-1][world_rank] = 0;
    if (world_rank != 0) {
        for (int i = 0; i < SIZE; i++) {
            result[layerCount-1][world_rank] += v[i];
        }
        result[layerCount-1][world_rank] = sigmoid(result[layerCount-1][world_rank]);
    } else {
        result[layerCount-1][world_rank] = 1;
    }
    float resultF = result[layerCount-1][world_rank] * recvbufT[layerCount][0];
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&resultF, &finalResult, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    target = 0.8;
    finalResult = sigmoid(finalResult);
    if (world_rank == 0) {
        printf("Result:%f\n", finalResult);
    }
    //backpropogation
    float e = error(target, finalResult);
    update(&recvbufT[layerCount][0], /*a[layer][i%SIZE]*/result[layerCount][world_rank], e, activationDifferential(finalResult));//update theta first then add to back
    //printf("2:%d,%f,%f,%f\n", world_rank,activationDifferential(finalResult)*recvbufT[layerCount][0], activationDifferential(finalResult), recvbufT[layerCount][0]);
    // if (i%(SIZE*SIZE)>0) { //thetas corresponding to the non-bias nodes
    //     back[layer][(i/SIZE)%SIZE-1][0] = activationDifferential(final)*theta[i];
    // }
    layerCount--;
    float b[SIZE]; //layer 0 (-1 in serial)
    float c = 1;
    if (world_rank!=0) { //last layer back prop
        c = activationDifferential(finalResult)*recvbufT[layerCount+1][0];
    }
    MPI_Allgather(&c,1,MPI_FLOAT,&b[0],1,MPI_FLOAT,MPI_COMM_WORLD);

    float pR = result[layerCount][world_rank];
    MPI_Allgather(&pR,1,MPI_FLOAT,&result[layerCount][0],1,MPI_FLOAT,MPI_COMM_WORLD);
    for (int w = 1; w < SIZE; w++) { //feature vector
        float back = 0;
        if (layerCount+1 == layers-1) { //final layer was last layer
            back = b[w];
        } else { //add all differentials
            for (int j = 1; j < SIZE; j++) {
                back += b[j];
            }
        }
        float last = activationDifferential(pR); //last z differential;
        update(&recvbufT[layerCount][w], recvbufF, e, last*b[w]); //update on features
        printf("%d,%f,%f,b%f\n", world_rank, back, recvbufT[layerCount][w], recvbufF);
    }
    //===================================================================================
    //forward propogation

    for (int i = 0; i < SIZE; i++) { //first set of activation functions
        pv[i] = recvbufT[layerCount][i] * recvbufF;
    }
    layerCount++;

    MPI_Barrier(MPI_COMM_WORLD);
    if (layers>2){ //middle layers
        for (int i = 1; i < layers-1; i++) {
            MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT,MPI_COMM_WORLD);
            result[layerCount][world_rank] = 0;
            if (world_rank != 0) {
                for (int i = 0; i < SIZE; i++) {
                    result[layerCount][world_rank] += v[i];
                }
                result[layerCount][world_rank] = sigmoid(result[layerCount][world_rank]);
            } else {
                result[layerCount][world_rank] = 1;
            }
            for (int i = 0; i < SIZE; i++) {
                pv[i] = recvbufT[layerCount][i] * result[layerCount][world_rank];
            }
            layerCount++;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT, MPI_COMM_WORLD);
    result[layerCount][world_rank] = 0;
    if (world_rank != 0) {
        for (int i = 0; i < SIZE; i++) {
            result[layerCount][world_rank] += v[i];
        }
        result[layerCount][world_rank] = sigmoid(result[layerCount][world_rank]);
    } else {
        result[layerCount][world_rank] = 1;
    }
    resultF = result[layerCount][world_rank] * recvbufT[layerCount][0];
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&resultF, &finalResult, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    target = 0.8;
    finalResult = sigmoid(finalResult);
    if (world_rank == 0) {
        printf("Result:%f\n", finalResult);
    }

    //MPI_Reduce(&pv, &result, 1, MPI_FLOAT, MPI_SUM, 0, layer_comm);

    // if(world_rank==0){ //serial part
    //     printf("%f\n", sigmoid(result));
    // }

    free(feature);
    free(theta);

	MPI_Finalize();
}
