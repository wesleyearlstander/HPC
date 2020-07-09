#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(){
#pragma omp parallel num_threads(4)
{
	#pragma omp single
	{
		printf("A(%d) ",omp_get_thread_num());
		#pragma omp task
			printf("race(%d) ",omp_get_thread_num());
		#pragma omp task
			printf("car(%d) ",omp_get_thread_num());
		#pragma omp task
			printf("is Fun to watch ");
		#pragma omp taskwait
		printf("is fun to watch ");
	}
	printf("is fun to watch ");
}
	printf("\n");
	return 0;
}

