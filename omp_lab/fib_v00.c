
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
int sFib(int n);
int pFib(int n);
int main(int argc, char *argv[]){
	int n, sRes, pRes, iter=10;
	double start_time, s_time=0.0, p_time=0.0;
	if(argc!=2){
		printf("Enter the number n.\n");
		exit(0);
	}
	n=atoi(argv[1]);
	start_time=omp_get_wtime();
	for(int k=0; k<iter;k++){
		sRes=sFib(n);
		s_time+=omp_get_wtime()-start_time;
	}
	s_time=s_time/iter;
	/*Call the parallel function pRes here in the similar
	way as sRes being called, and compare the performance.*/
	start_time=omp_get_wtime();
	for (int k=0; k<iter;k++) {
		#pragma omp parallel firstprivate(n)
		{
			#pragma omp single
				pRes=pFib(n);
		}
		p_time+=omp_get_wtime()-start_time;
	}
	p_time=p_time/iter;
	/* Uncomment the following if-else statement once you
	you completes the code */
	if(sRes==pRes)
		printf("The %dth Fibonacci number is: %d; s: %f, p: %f, speed up: %f\n", n, pRes,s_time,p_time,s_time/p_time);
	else
		printf("%d %d Error.\n", pRes, sRes);
	return 0;
}

int sFib(int n){
	int x, y;
	if(n<2)
		return n;
	else{
		x=sFib(n-1);
		y=sFib(n-2);
		return x+y;
	}
}
/* Parallelize the sFib() using OpenMP*/
int pFib(int n) {
	int x, y;
	if (n<2)
		return n;
	else if (n<35) {
		sFib(n);
	} else {
		#pragma omp task shared(x)
			x = pFib(n-1);
		#pragma omp task shared(y)
			y = pFib(n-2);
		#pragma omp taskwait
		return x+y;
	}
}
