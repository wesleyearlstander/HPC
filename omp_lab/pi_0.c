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
int main ()
{
	double start_time, run_time=0, pi;
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
	double pi, x, sum=0.0;
	for (int i=1;i<= num_steps; i++){
		x = (i-0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	return pi;
}



