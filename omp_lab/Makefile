INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc

all: hello hello_omp parallel_region pi pi_par

hello: hello.c
	$(CC) hello.c -o hello

hello_omp: hello_omp.c
	$(CC) hello_omp.c -o hello_omp $(OMPFLAG)

parallel_region: parallel_region.c
	$(CC) parallel_region.c -o parallel_region $(OMPFLAG)

pi: pi.c
	$(CC) pi.c -o pi $(OMPFLAG)

pi_par: pi_par.c
	$(CC) pi_par.c -o pi_par $(OMPFLAG)

clean:
	rm hello hello_omp parallel_region pi pi_par
