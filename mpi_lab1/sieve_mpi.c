#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
	int count;
	double elapsed_time;
	int first;
	int global_count;
	int high_value, low_value;
	int i, id, index, n, p;
	char *marked;
	int k, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Barrier(MPI_COMM_WORLD);

	// check if number n (for range 2~n) is entered from command line
	// if not, exit.
	if(argc != 2){
		if(!id) printf("Command line: %s <m>\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}

	// read n
	n = atoi(argv[1]);
	elapsed_time = -MPI_Wtime();
	// each process id computes its number range,
	// between low_value and high_value to mark.
	// id * (n-1)/p -- approximates the floor of id*n/p
	// the first prime number is 2, thus add 2 to shift all
	// low_values by 2, similarly for high_value
	low_value = 2 + id * (n-1)/p;
	// if integer division (n-1)/p is not exact,
	// the last process is allocated all the remaining numbers to mark.
	// the high_value for process id is 1 less than the low_value for process id+1,
	// hence, high_value = 2 + ((id + 1) * (n-1)/p - 1)
	if((n-1)%p && id==(p-1))
		high_value = n;
	else
		high_value = 2 + ((id + 1) * (n-1)/p - 1);
    // compute the size of numbers for each process.
	size = high_value - low_value + 1;

	// we want all the primes number sieves to be within the first
	// block to be processed by process 0.
	// We only need to sieve if k^2 > n, thus if the following condition
	// is true, then all k's we use will be within the first block owned by
	// process 0. Otherwise, exit.
	if((2 + size) < (int) sqrt( (double)n)){
		if(!id) printf("Too many processes");
		MPI_Finalize();
		exit(1);
	}


	marked = (char *)malloc(size);
	if(marked == NULL){
		printf("Cannot allocate enough memory\n");
		MPI_Finalize();
		exit(1);
	}
	// initialize the array elements as unmarked
	for(i = 0; i < size; i++)
		marked[i] = 0;

	if(!id)
		// only process 0 has this value
		index = 0;
	// starts from 2
	k = 2;
	do{
		// determine where to start marking for my portion
		if(k * k > low_value)
            // we use the local index for each process, the numbers
            // are stored in an array of size determined by
            // size = high_value - low_value + 1;
			first = k*k - low_value;
		else{
			if(!(low_value % k))
				first = 0;
			else
				first = k - (low_value % k);
		}
		// mark multiples of k in my portion
		for(i = first; i < size; i+=k)
			marked[i]=1;
		// find the next smallest unmarked number
		// in process 0's portion
		if(!id){
			while(marked[++index]);
			k = index + 2;
		}
		// broadcast the next k
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}while(k * k <= n);
	count = 0;
	// count the primes in my portion
	for(i=0; i<size; i++)
		if(!marked[i]) count++;
	// do a global reduction on the local counts
	MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	elapsed_time += MPI_Wtime();
	// process 0 to print the results to stdout
	if(!id){
		printf("%d primes are less than or equal to %d\n", global_count, n);
		printf("Total elapsed time: %10.6f\n", elapsed_time);
	}

	MPI_Finalize();
	return 0;
}
