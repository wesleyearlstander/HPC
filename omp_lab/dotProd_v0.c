#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 
#include <math.h>
float sDotProd(float *x, float *y, int N);
float pDotProd(float *x, float *y, int N);
void speedUp(int n, int numIter);
int main(int argc,char **argv)
{
	int n,i, numIter;
	float* x;
	float* y;
	float dotS, dotP;
	float t, sT, pT, sUp;
	float alpha;	
	if(argc==3){
		n=atoi(argv[1]);
		numIter=atoi(argv[2]);
	} 
	else{
		printf("Enter the size of the array and No of iteration.\n");
		exit(0);
	} 
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));
	
	//initialize
	srand(time(NULL)) ;
    for (i = 0; i < n; i++){
        x[i] = rand()/ RAND_MAX;
       	y[i] = rand()/ RAND_MAX;
    }
	//serial
	dotS=sDotProd(x, y, n);
	//parallel 
    dotP=pDotProd(x, y, n);	
	//verify the results from the serial,parallel versions 
	if(dotS!=dotP){
		printf("Verifiaction failed!\n");
		exit(0);}
	else{
		printf("Verifiaction passed!\n");
		speedUp(n, numIter);
	}	
	free(x);
	free(y);
	return 0;
}

float sDotProd(float *x, float *y, int N){
	int i;
	float dot=0.0; 
	for(i=0; i<N; i++){
		dot += x[i] * y[i];
	}
	return dot;	
}

//Complete the OpenMP parallelization of sDotProd() 
float pDotProd(float *x, float *y, int N){
	
}



void speedUp(int n, int numIter){
	float *x, *y, dotS, dotP;
	float sT=0.0, pT=0.0, t, sUp;
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));
	for(int k=0; k<numIter; k++){
		srand(time(NULL));
    	for (int i = 0; i < n; i++){
        	x[i] = rand()/ RAND_MAX;
       		y[i] = rand()/ RAND_MAX;
    	}
		//serial
		t=omp_get_wtime();
    	dotS=sDotProd(x, y, n);
		sT += omp_get_wtime()-t;
		//parallel 
		t=omp_get_wtime();
    	dotP=pDotProd(x, y, n);	
		pT += omp_get_wtime()-t;
	}
	sT /= numIter;
	pT /= numIter;
	sUp=sT/pT;
	printf("\n---====Problem size = %d====---\n", n);
    printf("The serial execution took %f seconds.\n", sT);
    printf("The parallel execution took %f seconds.\n", pT);
    printf("That's a speed up of %f.\n", sUp);
	free(x);
	free(y);
	return;
}

