/*
This program will numerically compute the integral of
                  4/(1+x*x)				  
from 0 to 1.  The value of this integral is pi. 
It uses the timer from the OpenMP runtime library
*/
#include <stdio.h>
#include <omp.h>
double compute_pi(double step);
static long num_steps = 1000000;
double step;
#define NUM_THREADS 4
int main (){
	
	double start_time, run_time=0.0, pi;
	int iter=10;
	step = 1.0/(double)num_steps;
	for(int i=0; i<iter; i++){
		start_time = omp_get_wtime();
		pi=compute_pi(step);
		run_time += omp_get_wtime() - start_time;
	}
	printf("\n pi with %ld steps is %f in %f seconds\n",num_steps,pi,run_time/iter);
}	  	  
double compute_pi(double step){
	int nthreads;
	double x, pi=0.0, sum[NUM_THREADS];
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, tthreads; 
		double x;
		tthreads = omp_get_num_threads();
		id = omp_get_thread_num();
		if(id==0) nthreads=tthreads;

		for (i=id, sum[id]=0.0;i< num_steps; i=i+tthreads){
			x = (i+0.5)*step;
			sum[id] = sum[id] + 4.0/(1.0+x*x);
		}
	}
	for(int i=0;i<nthreads;i++)			
			pi += step *sum[i];
	return  pi;
}




