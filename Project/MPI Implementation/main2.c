#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "metrics.h"
#define ALPHA 0.5
#define SIZE 14
#define TSIZE 242
#define TESTSIZE 61


float sigmoid(float x){
     return 1.0 / (1.0 + exp(-x));
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
    //return pow(actual-target,2)/2.0; //cross entropy error
}

void update(float* weight, float p, float error, float partial) {
    *weight = *weight - ALPHA*error*partial*p;
}
//serial forward propagation
float neuralNetSerial(float* feature, float* theta, int layers, float bias) {
    float a[layers-1][SIZE];
    a[0][0] = bias;
    for (int i = 1; i < SIZE; i++) { //first layer
        a[0][i] = activation(&theta[i], feature, SIZE);
    }
    int finalLayer = layers-1;
    if (layers>2) { //more than 2 layers
        for (int l = 1; l < finalLayer; l++) {
            a[l][0] = bias;
            for (int i = 1; i < SIZE; i++) {
                a[l][i] = activation(&theta[i + l*SIZE*SIZE], a[l-1], SIZE);
            }
        }
    }
    float out = 0;
    for (int i = 0; i < SIZE; i++) { //last layer
        out += a[finalLayer-1][i] * theta[finalLayer*SIZE*SIZE + i*SIZE];
        //printf("%d, %f,%f,%f\n", i, a[finalLayer-1][i] * theta[finalLayer*SIZE*SIZE + i*SIZE],theta[finalLayer*SIZE*SIZE + i*SIZE],a[finalLayer-1][i]);
    }
    //printf("%f\n", out);
    float final = sigmoid(out); //forward propogate
    return final;
}
//serial learning
float learnSerial(float* feature, float* theta, int layers, float bias, int max_epoch, float epsilon, float* target, int size) {
    int epoch = 0;
    float final = 0;
    float previous = 0;
    while (1) {
        for (int value = 0; value < TSIZE; value++) {
            //forward
            float a[layers-1][SIZE];
            a[0][0] = bias;
            for (int i = 1; i < SIZE; i++) {
                a[0][i] = activation(&theta[i], &feature[value*SIZE], SIZE);
            }
            int finalLayer = layers-1;
            if (layers>2) {
                for (int l = 1; l < finalLayer; l++) {
                    a[l][0] = bias;
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
            if(fabs(final - target[value])<epsilon) break; //below error requirement
            //===========================================================
            //back propogation
            //learn
            int layer = finalLayer-1;
            float e = error(target[value], final);
            float back[layers-1][SIZE-1][SIZE-1];
            for (int i = layers*SIZE*SIZE-1; i>=0;i--) {
                if (layer != -1) {
                    if (layer == finalLayer-1) { //last layer (can be 0(2layers) or 1(3layers))
                        if (i%SIZE==0) { //works out the thetas corresponding to the final node
                            if (i%(SIZE*SIZE)>0) { //thetas corresponding to the non-bias nodes
                                back[layer][(i/SIZE)%SIZE-1][0] = theta[i];
                            }
                            update(&theta[i], a[layer][i%(SIZE*SIZE)/SIZE], e, 1.0);//update theta first then add to back
                        }
                    }
                    else { //layer 0 to finalLayer-1
                        if (i%SIZE>0) { //ignores the bias node weights for next layer
                            float b = 0;
                            if (layer+1 == finalLayer-1) { //final layer was last layer
                                b = back[layer+1][i%SIZE-1][0];
                            } else { //add all differentials
                                for (int w = 0; w < SIZE-1; w++) {
                                    b += back[layer+1][i%SIZE-1][w];
                                }
                            }
                            float last = activationDifferential(a[layer+1][i%SIZE]); //last z differential;
                            back[layer][(int)(i%(SIZE*SIZE)/SIZE)-1][i%SIZE-1] = last*theta[i]*b; //update next layer differentials
                            update(&theta[i], a[layer][i%(SIZE*SIZE)/SIZE], e, last*b);
                        }
                    }
                }
                else { //first layer (-1)
                    if (i%SIZE>0) { //ignores the bias node weights for next layer
                        float b = 0;
                        if (layer+1 == finalLayer-1) { //final layer was last layer
                            b = back[layer+1][i%SIZE-1][0];
                        } else { //add all differentials
                            for (int w = 0; w < SIZE-1; w++) {
                                b += back[layer+1][i%SIZE-1][w];
                            }
                        }
                        float last = activationDifferential(a[layer+1][i%SIZE]); //last z differential;
                        update(&theta[i], feature[value*SIZE + (int)(i/SIZE)], e, last*b); //update on features
                    }
                }
                if (i%(SIZE*SIZE)==0) { //last theta of each layer
                    layer--; //move back a layer
                }
            }
        }
        epoch++;
        if(epoch >= max_epoch) break; //max epoch reached
    }
    return neuralNetSerial(feature, theta, layers, bias);
}
//MPI approach one forward propagation
float neuralNetMPI(float * feature, float * theta, int layers, float bias) {
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    float recvbufT[layers][SIZE];
    float recvbufF;
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
    if (world_rank == 0) recvbufF = bias; //bias
    for (int i = 1; i < SIZE; i++) { //first set of activation functions
        pv[i] = recvbufT[layerCount][i] * recvbufF;
    }
    layerCount++;

    if (layers>2){ //middle layers
        for (int l = 1; l < layers-1; l++) {
            MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT,MPI_COMM_WORLD);
            result[layerCount-1][world_rank] = 0;
            if (world_rank != 0) {
                for (int i = 0; i < SIZE; i++) {
                    result[layerCount-1][world_rank] += v[i];
                }
                result[layerCount-1][world_rank] = sigmoid(result[layerCount-1][world_rank]);
            } else {
                result[layerCount-1][world_rank] = bias;
            }
            for (int i = 1; i < SIZE; i++) {
                pv[i] = recvbufT[layerCount][i] * result[layerCount-1][world_rank];
            }
            layerCount++;
        }
    }

    MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT, MPI_COMM_WORLD);
    result[layerCount-1][world_rank] = 0;
    if (world_rank != 0) { //final layer
        for (int i = 0; i < SIZE; i++) {
            result[layerCount-1][world_rank] += v[i];
        }
        result[layerCount-1][world_rank] = sigmoid(result[layerCount-1][world_rank]);
    } else {
        result[layerCount-1][world_rank] = bias;
    }
    float resultF = result[layerCount-1][world_rank] * recvbufT[layerCount][0];
    MPI_Allreduce(&resultF, &finalResult, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    finalResult = sigmoid(finalResult);
    return finalResult;
}
//MPI approach two forward propagation
float neuralNetMPIFaster(float* feature, float* theta, int layers, float bias) {
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    float result[layers-1][SIZE];
    float finalResult = 0;
    int subSize = ceil((float)SIZE/(float)world_size);
    int layerCount = 0;
    for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //first set of activation functions
        if (i==0) result[layerCount][i] = bias;
        else result[layerCount][i] = activation(&theta[i], feature, SIZE);
    }

    int currentSize = 0;
    for (int i = 0; i < world_size && world_size >1; i++) {
        if (SIZE-currentSize >= subSize)
            MPI_Bcast(&result[layerCount][subSize*i], subSize, MPI_FLOAT, i, MPI_COMM_WORLD); //synchronize
        else if (SIZE-currentSize > 0)
            MPI_Bcast(&result[layerCount][subSize*i], SIZE-currentSize, MPI_FLOAT, i, MPI_COMM_WORLD); //in case of remainder
        currentSize += subSize;
    }
    layerCount++;

    if (layers>2){ //middle layers
        for (int l = 1; l < layers-1; l++) {
            for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //activation functions
                if (i==0) result[layerCount][i] = bias;
                else result[layerCount][i] = activation(&theta[i + layerCount*SIZE*SIZE], result[layerCount-1], SIZE);
            }
            currentSize = 0;
            for (int i = 0; i < world_size && world_size >1; i++) {
                if (SIZE-currentSize >= subSize)
                    MPI_Bcast(&result[layerCount][subSize*i], subSize, MPI_FLOAT, i, MPI_COMM_WORLD); //synchronize
                else if (SIZE-currentSize > 0)
                    MPI_Bcast(&result[layerCount][subSize*i], SIZE-currentSize, MPI_FLOAT, i, MPI_COMM_WORLD); //in case of remainder
                currentSize += subSize;
            }
            layerCount++;
        }
    }

    finalResult = 0;
    for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //last activation function
        finalResult += theta[layerCount*SIZE*SIZE + i*SIZE] * result[layerCount-1][i];
    }

    float final = 0;
    MPI_Allreduce(&finalResult, &final, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    final = sigmoid(final);
    return final;
}
//MPI approach two learning
float learnMPIFaster (float* feature, float* theta, int layers, float bias, int max_epoch, float epsilon, float* target) {
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    float finalResult = 0;
    int subSize = ceil((float)SIZE/(float)world_size);
    int backSize = ceil(((float)SIZE*SIZE)/(float)world_size);
    int epoch = 0;
    while(1) {
        for (int value = 0; value < TSIZE; value++) {
            int layerCount = 0;
            float result[layers][SIZE];
            for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //first set of activation functions
                if (i==0) result[layerCount+1][i] = bias;
                else result[layerCount+1][i] = activation(&theta[i], &feature[value*SIZE], SIZE);
            }

            int currentSize = 0;
            for (int i = 0; i < world_size && world_size >1; i++) {
                if (SIZE-currentSize >= subSize)
                    MPI_Bcast(&result[layerCount+1][subSize*i], subSize, MPI_FLOAT, i, MPI_COMM_WORLD); //synchronize
                else if (SIZE-currentSize > 0)
                    MPI_Bcast(&result[layerCount+1][subSize*i], SIZE-currentSize, MPI_FLOAT, i, MPI_COMM_WORLD); //in case of remainder
                currentSize += subSize;
            }
            layerCount++;

            if (layers>2){ //middle layers
                for (int l = 1; l < layers-1; l++) {
                    for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //activation functions
                        if (i==0) result[layerCount+1][i] = bias;
                        else result[layerCount+1][i] = activation(&theta[i + layerCount*SIZE*SIZE], result[layerCount], SIZE);
                    }
                    currentSize = 0;
                    for (int i = 0; i < world_size && world_size >1; i++) {
                        if (SIZE-currentSize >= subSize)
                            MPI_Bcast(&result[layerCount+1][subSize*i], subSize, MPI_FLOAT, i, MPI_COMM_WORLD); //synchronize
                        else if (SIZE-currentSize > 0)
                            MPI_Bcast(&result[layerCount+1][subSize*i], SIZE-currentSize, MPI_FLOAT, i, MPI_COMM_WORLD); //in case of remainder
                        currentSize += subSize;
                    }
                    layerCount++;
                }
            }

            finalResult = 0;
            for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) {
                finalResult += theta[layerCount*SIZE*SIZE + i*SIZE] * result[layerCount][i];
            }

            float final = 0;
            MPI_Allreduce(&finalResult, &final, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            final = sigmoid(final); //last activation function
            if(fabs(final - target[value])<epsilon) break; //below error requirement
            //===================================================================
            //backpropogation
            float e = error(target[value], final);
            float b[layers-1][SIZE][SIZE]; //layer 0 (-1 in serial)
            for (int i = subSize*(world_rank); i < subSize*(world_rank+1) && i < SIZE; i++) { //first set of activation functions
                if (i!=0)
                    b[layerCount-1][i][0] = theta[layerCount*SIZE*SIZE + i*SIZE];
                update(&theta[layerCount*SIZE*SIZE + i*SIZE],result[layerCount][i], e, 1.0);//add back first then update theta
            }

            currentSize = 0;
            for (int i = 0; i < world_size && world_size >1; i++) {
                if (SIZE-currentSize >= subSize) {
                    MPI_Datatype t_column;
                    MPI_Type_vector(subSize, 1, SIZE, MPI_FLOAT, &t_column);
                    MPI_Type_commit(&t_column);
                    MPI_Bcast(&b[layerCount-1][subSize*i][0], 1, t_column, i, MPI_COMM_WORLD); //synchronize
                    MPI_Bcast(&theta[layerCount*SIZE*SIZE + i*SIZE*subSize], SIZE*subSize, MPI_FLOAT, i, MPI_COMM_WORLD);
                } else if (SIZE-currentSize > 0){
                    MPI_Datatype t_column;
                    MPI_Type_vector(SIZE-currentSize, 1, SIZE, MPI_FLOAT, &t_column);
                    MPI_Type_commit(&t_column);
                    MPI_Bcast(&b[layerCount-1][subSize*i][0], 1, t_column, i, MPI_COMM_WORLD); //synchronize
                    MPI_Bcast(&theta[layerCount*SIZE*SIZE + i*SIZE*subSize], SIZE*(SIZE-currentSize), MPI_FLOAT, i, MPI_COMM_WORLD);
                }
                currentSize += subSize;
            }

            if (layers>2) {
                for (int l = 1; l < layers-1; l++) {
                    for (int i = backSize*(world_rank) + SIZE*SIZE*(layerCount-1); i < backSize*(world_rank+1) + SIZE*SIZE*(layerCount-1) && i < SIZE*SIZE*(layerCount); i++) { //first set of activation functions
                        if (i%SIZE>0) {
                            float back = 0;
                            if (layerCount+1 == layers) { //final layer was last layer
                                back = b[layerCount-1][i%SIZE][0];
                            } else { //add all differentials
                                for (int j = 1; j < SIZE; j++) {
                                    back += b[layerCount-1][i%SIZE][j];
                                }
                            }
                            float last = activationDifferential(result[layerCount][i%SIZE]); //last z differential;
                            b[layerCount-2][(int)(i%(SIZE*SIZE)/SIZE)][i%SIZE] = last*theta[i]*back;
                            update(&theta[i], result[layerCount-1][(int)(i%(SIZE*SIZE)/SIZE)], e,  back * last); //update on features
                        }
                    }

                    layerCount--;
                    currentSize = 0;
                    for (int i = 0; i < world_size && world_size >1; i++) {
                        if ((backSize*world_size)-currentSize >= backSize) {
                            MPI_Datatype row;
                            MPI_Type_vector(1, backSize, backSize, MPI_FLOAT, &row);
                            MPI_Type_commit(&row);
                            MPI_Bcast(&b[layerCount-1][(backSize*i)/SIZE][(backSize*i)%SIZE], 1, row, i, MPI_COMM_WORLD); //synchronize
                            MPI_Bcast(&theta[layerCount*SIZE*SIZE + i*backSize], 1, row, i, MPI_COMM_WORLD);
                        } else if ((backSize*world_size)-currentSize > 0){ //remainder
                            MPI_Datatype row;
                            MPI_Type_vector(1, (backSize*world_size)-currentSize, (backSize*world_size)-currentSize, MPI_FLOAT, &row);
                            MPI_Type_commit(&row);
                            MPI_Bcast(&b[layerCount-1][(backSize*i)/SIZE][(backSize*i)%SIZE], 1, row, i, MPI_COMM_WORLD); //synchronize
                            MPI_Bcast(&theta[layerCount*SIZE*SIZE + i*SIZE*subSize],1, row, i, MPI_COMM_WORLD);
                        }
                        currentSize += backSize;
                    }
                }
            }

            for (int i = backSize*(world_rank); i < backSize*(world_rank+1) && i < SIZE*SIZE; i++) { //first set of activation functions
                if (i%SIZE>0) {
                    float back = 0;
                    if (layerCount+1 == layers) { //final layer was last layer
                        back = b[layerCount-1][i%SIZE][0];
                    } else { //add all differentials
                        for (int j = 1; j < SIZE; j++) {
                            back += b[layerCount-1][i%SIZE][j];
                        }
                    }
                    float last = activationDifferential(result[layerCount][i%SIZE]); //last z differential;
                    update(&theta[i], feature[value*SIZE + i/SIZE], e,  back * last); //update on features
                }
            }

            currentSize = 0;
            for (int i = 0; i < world_size && world_size >1; i++) {
                if ((backSize*world_size)-currentSize >= backSize) {
                    MPI_Bcast(&theta[i*backSize], backSize, MPI_FLOAT, i, MPI_COMM_WORLD);
                } else if ((backSize*world_size)-currentSize > 0){
                    MPI_Bcast(&theta[i*backSize], (backSize*world_size)-currentSize, MPI_FLOAT, i, MPI_COMM_WORLD);
                }
                currentSize += backSize;
            }

        }
        epoch++;
        if(epoch >= max_epoch) break; //max epoch reached
    }

    return neuralNetMPIFaster(feature, theta, layers, bias);
}
//MPI approach one learning
float learnMPI(float* feature, float* theta, int layers, float bias, int max_epoch, float epsilon, float* target) {
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    float recvbufT[layers][SIZE];
    float recvbufF;
    //distribute theta
    for (int i = 0; i < layers; i++) {
        MPI_Scatter(&theta[i*SIZE*SIZE], SIZE, MPI_FLOAT, &recvbufT[i][0], SIZE, MPI_FLOAT,0,MPI_COMM_WORLD);
    }

    float pv[SIZE];
    float v[SIZE];
    float result[layers-1][SIZE];
    float finalResult = 0;
    float previous = 0;

    int layerCount = 0;
    //distribute feature vector


    int epoch = 0;
    while(1) {
        for (int value = 0; value < TSIZE; value++) {
            MPI_Scatter(&feature[value*SIZE], 1, MPI_FLOAT, &recvbufF, 1, MPI_FLOAT,0,MPI_COMM_WORLD);
            if (world_rank == 0) recvbufF = bias; //bias

            for (int i = 1; i < SIZE; i++) { //first set of activation functions
                pv[i] = recvbufT[layerCount][i] * recvbufF;
            }
            layerCount++;

            if (layers>2){ //middle layers
                for (int l = 1; l < layers-1; l++) {
                    MPI_Alltoall(&pv, 1, MPI_FLOAT, &v, 1, MPI_FLOAT,MPI_COMM_WORLD);
                    result[layerCount-1][world_rank] = 0;
                    if (world_rank != 0) {
                        for (int i = 0; i < SIZE; i++) {
                            result[layerCount-1][world_rank] += v[i];
                        }
                        result[layerCount-1][world_rank] = sigmoid(result[layerCount-1][world_rank]);
                    } else {
                        result[layerCount-1][world_rank] = bias;
                    }
                    for (int i = 1; i < SIZE; i++) {
                        pv[i] = recvbufT[layerCount][i] * result[layerCount-1][world_rank];
                    }
                    layerCount++;
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
                result[layerCount-1][world_rank] = bias;
            }
            float resultF = result[layerCount-1][world_rank] * recvbufT[layerCount][0];
            MPI_Allreduce(&resultF, &finalResult, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            finalResult = sigmoid(finalResult);
            //testing criteria
            //if (finalResult == previous) break; //test convergence
            previous = finalResult;
            if(fabs(finalResult - target[value])<epsilon) break; //below error requirement
            //===================================================================
            //backpropogation
            float e = error(target[value], finalResult);
            layerCount--;
            float pR;
            if (layers>2) {
                pR = result[layerCount][world_rank]; //gather values for layer and distribute to all nodes
                MPI_Allgather(&pR,1,MPI_FLOAT,&result[layerCount][0],1,MPI_FLOAT,MPI_COMM_WORLD);
            }
            float b[layers-1][SIZE-1][SIZE]; //layer 0 (-1 in serial)
            float c = 1;
            if (world_rank!=0) { //last layer back prop
                c = recvbufT[layerCount+1][0]; //last layer b
            }
            MPI_Allgather(&c,1,MPI_FLOAT,&b[layerCount][0][0],1,MPI_FLOAT,MPI_COMM_WORLD);
            update(&recvbufT[layerCount+1][0],result[layerCount][world_rank], e, 1.0);//add back first then update theta
            //printf("%d, %f, %f, %f, %f\n", world_rank,recvbufT[layerCount+1][0], result[layerCount][world_rank], e,activationDifferential(finalResult));
            if (layers>2) {
                for (int i = 1; i < layers-1; i++) {
                    pR = result[layerCount-1][world_rank]; //gather values for lower layer
                    MPI_Allgather(&pR,1,MPI_FLOAT,&result[layerCount-1][0],1,MPI_FLOAT,MPI_COMM_WORLD);

                    for (int w = 1; w < SIZE; w++) { //feature vector
                        float back = 0;
                        if (layerCount+1 == layers-1) { //final layer was last layer
                            back = b[layerCount][0][w];
                        } else { //add all differentials
                            for (int j = 1; j < SIZE; j++) {
                                back += b[layerCount][j][w];
                            }
                        }
                        float last = activationDifferential(result[layerCount][w]); //last z differential;
                        if (world_rank!=0) { //last layer back prop ignoring bias term
                            c = last*back*recvbufT[layerCount][w]; //last layer b
                        }
                        MPI_Allgather(&c,1,MPI_FLOAT,&b[layerCount-1][w][0],1,MPI_FLOAT,MPI_COMM_WORLD);
                        update(&recvbufT[layerCount][w], result[layerCount-1][world_rank], e, last*back); //update on features
                    }

                    layerCount--;
                }
            }

            if (layerCount+1==layers-1) {
                pR = result[layerCount][world_rank]; //gather values for layer
                MPI_Allgather(&pR,1,MPI_FLOAT,&result[layerCount][0],1,MPI_FLOAT,MPI_COMM_WORLD);
            }

            for (int w = 1; w < SIZE; w++) { //feature vector
                float back = 0;
                if (layerCount+1 == layers-1) { //final layer was last layer
                    back = b[layerCount][0][w];
                } else { //add all differentials
                    for (int j = 1; j < SIZE; j++) {
                        back += b[layerCount][j][w];
                    }
                }
                float last = activationDifferential(result[layerCount][w]); //last z differential;
                update(&recvbufT[layerCount][w], recvbufF, e, last*back); //update on features
                //printf("%d, %f, %f, %f, %f\n", world_rank, recvbufT[layerCount][w], recvbufF, e, last*back);
            }
        }
        epoch++;
        if(epoch >= max_epoch) break; //max epoch reached
    }

    for (int i = 0; i < layers; i++) { //return theta back to theta array
        MPI_Gather(&recvbufT[i][0], SIZE, MPI_FLOAT, &theta[i*SIZE*SIZE], SIZE, MPI_FLOAT,0,MPI_COMM_WORLD);
    }

    return neuralNetMPI(feature, theta, layers, bias);
}

int prediction (float value) {
    if (value >= 0.5) return 1;
    else return 0;
}

int calculations(int layers) {
    int i = SIZE;
    int forward = ((i-1)*(i+(i-1)+3))*(layers-1) + (i+(i-1)+3) + i*4 + i*(i-1)*8;
    int back = (((i-1)*6+6)*(i*(i-1)))*(layers-2);
    return forward+back;
}

float serialAccuracy (float* feature, float* theta, int layers, float bias,float* target, int size) {
    int accurateSample = 0;
    for (int i = 0; i < size; i++) {
        float output = neuralNetSerial(&feature[i*SIZE], theta, layers, bias);
        if (prediction(output) == target[i])
            accurateSample++;
    }
    return (float)accurateSample/(float)size;
}

float MPIAccuracy (float* feature, float* theta, int layers, float bias,float* target, int size) {
    int accurateSample = 0;
    for (int i = 0; i < size; i++) {
        if (prediction(neuralNetMPI(&feature[i*SIZE], theta, layers, bias)) == target[i])
            accurateSample++;
    }
    return (float)accurateSample/(float)size;
}

float MPIAccuracyImproved(float* feature, float* theta, int layers, float bias,float* target, int size, int world_rank) {
    int accurateSample = 0;
    int samplesOG = size/SIZE;
    int samples = samplesOG;
    if (world_rank == SIZE-1) {
        if (size%SIZE != 0) {
            samples+= size%SIZE;
        }
    }
    for (int i = 0; i < samples; i++) {
        if (prediction(neuralNetSerial(&feature[world_rank*SIZE + i*SIZE], theta, layers, bias)) == target[world_rank*samplesOG + i])
            accurateSample++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int total = 0;
    MPI_Reduce(&accurateSample, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return (float)total/(float)size;
}

int main(int argc, char *argv[]){
    int world_size, world_rank;
	int sendcount, recvcount, source;
    int max_epoch = 1;
    const float epsilon = pow(10,-7);
    int layers = 3;
    if (argc > 1)
        layers = atoi(argv[1]);
    if (argc > 2)
        max_epoch = atoi(argv[2]);
    float bias = 1;

    Metrics* metricsPre = malloc(sizeof(Metrics));
    Metrics* metrics = malloc(sizeof(Metrics));

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    metricsPre->processes = world_size;
    metrics->processes = world_size;

    float* feature = malloc(sizeof(float)*SIZE*TSIZE);//242 values
    float* target = malloc(sizeof(float)*TSIZE);

    float* featureTest = malloc(sizeof(float)*SIZE*TESTSIZE);//242 values
    float* targetTest = malloc(sizeof(float)*TSIZE);

    if (world_rank == 0) { //Read in training file
        FILE *fp = fopen("training.csv", "r");
        char buf[1024];

        int row_count = -2;

        while (fgets(buf, 1024, fp)) {
            row_count++;
            if (row_count==-1) continue;
            int columnCount = 0;
            char *field = strtok(buf, ",");
            while (field) {
                if (columnCount == 0) {
                    feature[row_count*SIZE + columnCount] = bias; //bias
                } else if (columnCount < SIZE && columnCount > 0) {
                    feature[row_count*SIZE + columnCount] = atof(field);
                } else if (columnCount == SIZE) {
                    target[row_count] = atof(field);
                }

                field = strtok(NULL, ",");
                columnCount++;
            }

        }

        fclose(fp);
        FILE *fpt = fopen("testing.csv", "r");
        buf[1024];

        row_count = -2;

        while (fgets(buf, 1024, fpt)) {
            row_count++;
            if (row_count==-1) continue;
            int columnCount = 0;
            char *field = strtok(buf, ",");
            while (field) {
                if (columnCount == 0) {
                    featureTest[row_count*SIZE + columnCount] = bias; //bias
                } else if (columnCount < SIZE && columnCount > 0) {
                    featureTest[row_count*SIZE + columnCount] = atof(field);
                } else if (columnCount == SIZE) {
                    targetTest[row_count] = atof(field);
                }

                field = strtok(NULL, ",");
                columnCount++;
            }

        }

        fclose(fpt);
    }

    MPI_Bcast(feature, SIZE*TSIZE, MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Bcast(target, TSIZE, MPI_FLOAT,0, MPI_COMM_WORLD);

    for (int a = 0; a < 1; a++) {
        float* thetaOriginal = malloc(sizeof(float)*SIZE*SIZE*layers);
        float* theta = malloc(sizeof(float)*SIZE*SIZE*layers);

        // if (world_rank ==0) {
        //     if (world_size != SIZE) {
        //         fprintf(f,"Make processes = %d\n", SIZE);
        //         return-1;
        //     }
        // }
        srand(time(NULL));
        if(world_rank==0){ //generate initial theta and features
            for (int l = 0; l < layers; l++) {
                for (int j = 0; j < SIZE; j++) {
                    for (int i = 1; i <= SIZE; i++) { //initialize weights randomly for stochastic descent
                        thetaOriginal[l*SIZE*SIZE + j*SIZE + (i-1)] = (float)rand()/(float)RAND_MAX;
                        //printf("%d,%f\n", l*SIZE*SIZE + j*SIZE + (i-1), thetaOriginal[l*SIZE*SIZE + j*SIZE + (i-1)]);
                    }
                }
            }
            memcpy(theta, thetaOriginal, sizeof(float)*SIZE*SIZE*layers);
            // double time = clock();
            // float ans = neuralNetSerial(feature,theta,layers,bias);
            // double end = clock();
            // metricsPre->serialTime = ((end - time)/(double)CLOCKS_PER_SEC);
            // metricsPre->result = ans;
            // printf("Serial_pre:%f,%f\n", metricsPre->serialTime ,ans); //output before training
            //time = clock();
            //float sAccuracy = serialAccuracy(feature,theta,layers,bias,target,TSIZE);
            //printf("SerialAccuracyTime:%f,%f\n", ((float)(clock() - time)/CLOCKS_PER_SEC), sAccuracy);
            // fprintf(f,"Serial_pre:%f\n", neuralNetSerial(feature,theta,layers,bias)); //output before training
            // fprintf(f,"SerialTime:%f\n", ((float)(clock() - time)/CLOCKS_PER_SEC));

        }

        MPI_Bcast(theta, layers*SIZE*SIZE, MPI_FLOAT,0, MPI_COMM_WORLD);


        if (world_rank==0) { //learning
            float time = clock();
            float serialLearnt = learnSerial(feature,theta,layers,bias,max_epoch,epsilon,target, TSIZE);
            float end = clock();
            metrics->serialTime = (end - time)/(double)CLOCKS_PER_SEC;
            printf("SerialTime:%f\n", metrics->serialTime);
            memcpy(theta, thetaOriginal, sizeof(float)*SIZE*SIZE*layers);
        }


        MPI_Barrier(MPI_COMM_WORLD);
        double time = MPI_Wtime();
        float prePar;
        if (world_size == SIZE)
            prePar = learnMPI(feature, theta, layers,bias, max_epoch, epsilon, target);
        double end = MPI_Wtime();
        //float parAccuracy = MPIAccuracy(feature, theta, layers, bias, target, TSIZE);

        if (world_rank == 0 && world_size == SIZE) {
            printf("ParallelTime:%f\n", end - time);
            metrics->parallelTime = end - time;
            float parAccuracyBetter = serialAccuracy(featureTest, theta, layers, bias, targetTest, TESTSIZE);
            printf("Test accuracy:%f%%\n", parAccuracyBetter*100);
            // fprintf(f,"ParallelTime:%f\n", MPI_Wtime() - time);
            // fprintf(f,"MPI_pre:%f\n",prePar);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        time = MPI_Wtime();
        float finalResult = learnMPIFaster(feature, theta, layers,bias, max_epoch, epsilon, target);
        end = MPI_Wtime();
        if (world_rank == 0 && world_size != SIZE) {
            metrics->parallelTime = end-time;
            // RunTestsOnly(metrics);
            printf("ParallelTime:%f\n", end - time);
            float parAccuracyBetter = serialAccuracy(featureTest, theta, layers, bias, targetTest, TESTSIZE);
            printf("Accuracy of ANN against testing data:%f%%\n", parAccuracyBetter*100);
            // printf("efficiency:%f\n", metrics->efficiency);

            // fprintf(f,"Result:%f\n", finalResult);

        }
        free(theta);
        free(thetaOriginal);
    }

    if (world_rank == 0) {
        // metrics->serialTime = metrics->serialTime;
        // metrics->parallelTime = metrics->parallelTime;
        RunTestsOnly(metrics);
        FILE *f = fopen("output.txt", "a");
        FILE *f2 = fopen("output2.txt", "a");
        fprintf(f, "%d,%d,%d,%f\n", world_size, layers, max_epoch, metrics->speedup);
        fprintf(f2, "%d,%d,%d,%f\n", world_size, layers, max_epoch, metrics->serialTime);
        printf("Speedup:%f\n", metrics->speedup);
        printf("Throughput:%f\n", abs(calculations(layers)*TSIZE*max_epoch)/(metrics->parallelTime * pow(10,9)));
        fclose(f);
        fclose(f2);
    }

    //MPI_Reduce(&pv, &result, 1, MPI_FLOAT, MPI_SUM, 0, layer_comm);

    // if(world_rank==0){ //serial part
    //     printf("%f\n", sigmoid(result));
    // }
    free(featureTest);
    free(targetTest);
    free(feature);
    free(target);
    free(metricsPre);
    free(metrics);
	MPI_Finalize();
}
