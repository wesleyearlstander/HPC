#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 
#include <math.h>
int sSaxpy(float alpha, float *x, float *y, float *zS, int N);
int pSaxpy(float alpha, float *x, float *y, float *zP, int N);
int vectorEquil(float *vecA, float *vecB, int n);
int main(int argc,char **argv)
{
	int n,i;
	float* x;
	float* y;
	float* zS;
	float* zP;
    float t, sT, pT, sUp;
	float alpha;	
	if(argc==3){
		n=atoi(argv[1]);
		alpha=atoi(argv[2]);
	} 
	else{
		printf("Enter the size of the array and alpah value\n");
		exit(0);
	} 
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));
	zS = (float*)malloc(n*sizeof(float));
	zP = (float*)malloc(n*sizeof(float));
	srand(time(NULL)) ;
	//initialize
    for (i = 0; i < n; i++){
        x[i] = rand()/ RAND_MAX;
       	y[i] = rand()/ RAND_MAX;
    }
	//serial
	t=omp_get_wtime();
    sSaxpy (alpha, x, y, zS, n);
	sT=omp_get_wtime()-t;
	//parallel 
	t=omp_get_wtime();
    pSaxpy (alpha, x, y, zP, n);	
	pT=omp_get_wtime()-t;
	//verify the results from the serial,parallel versions 
	if(vectorEquil(zS,zP,n)){
		printf("Verifiaction failed!\n");
		exit(0);}
	else 
		printf("Verifiaction passed!\n");
	sUp=sT/pT;
	printf("\n---====Problem size = %d====---\n", n);
    printf("The serial execution took %f seconds.\n", sT);
    printf("The parallel execution took %f seconds.\n", pT);
    printf("That's a speed up of %f.\n", sUp);
	free(x);
	free(y);
	free(zS);
	free(zP);
	return 0;
}

int sSaxpy(float alpha, float *x, float *y, float *zS, int N){
	int i; 
	for(i=0; i<N; i++){
		zS[i]= y[i] + alpha*x[i];
	}
	return 0;	
}
//Complete the parallel version of saxpy using OpenMP
int pSaxpy(float alpha, float *x, float *y, float *zP, int N){
	
	return 0;	
}
int vectorEquil(float *vecA, float *vecB, int n){
	int bad=0;
	for(int i=0; i<n; i++){
		if(vecA[i] != vecB[i])
			bad++;
	}
	return bad;
}

