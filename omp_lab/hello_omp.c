#include <stdio.h>
#include <omp.h>
int main()
{
	//Parallel region with default number of threads
	#pragma omp parallel// num_threads(4)
	//Start of the parallel region
	{
		//Runtime library function to return a thread number
		int ID = omp_get_thread_num();
		printf("Hello World! (Thread %d)\n", ID);
	}//End of the parallel region
}
 

