#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
void simple(int n, float *a, float *b);
int main(int argc, char** argv){
	int n=16;
	float *a, *b;
	/*if(argc!=2){
		printf("Enter n (n < 24) \n");
		exit(0);
	}
	n=atoi(argv[1]);*/
	a=(float*)malloc(n*sizeof(float));
	b=(float*)malloc(n*sizeof(float));
	srand(time(NULL));
	for(int i=0; i<n; i++){
		a[i]=(float)rand()/RAND_MAX;
		b[i]=(float)rand()/RAND_MAX;
	}
	simple(n,a,b);
	free(a);
	free(b);
	return 0;
}
void simple(int n, float *a, float *b)
{
    int i;
//add a schedule clause and change the setting of its value to see the effects
#pragma omp parallel for 
    for (i=1; i<n; i++){ /* i is private by default */
        b[i] = (a[i] + a[i-1]) / 2.0;
		printf("I am thread: %d at %d computed b[%d]=%f \n",omp_get_thread_num(), i, i, b[i]);
	}
}



