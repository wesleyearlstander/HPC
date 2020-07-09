
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#define MSG_SZ 128

int main(int argc, char *argv[]){
	int nproces, myrank, next, prev, tag=1;
	char token[MSG_SZ];	

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproces);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	if(myrank == 0)
		prev = nproces - 1;
	else
		prev = myrank - 1;
	
	if(myrank == (nproces - 1))  
		next = 0;
	else 
		next = myrank + 1;	
	if(myrank == 0)
		strcpy(token, "Hello!");
	MPI_Send(token, MSG_SZ, MPI_CHAR, next, tag, MPI_COMM_WORLD);
	MPI_Recv(token, MSG_SZ, MPI_CHAR, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	printf("Process %d received token %s from process %d.\n", myrank, token, prev);
	
	MPI_Finalize();
	return 0;
}
