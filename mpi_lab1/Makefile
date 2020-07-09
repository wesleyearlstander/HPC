P1 = mpi_ring_v2
P2 = ring2
CC = mpicc

all: $(P2) ring4 sieve_mpi

$(P2): $(P1).c 
	$(CC) $(P1).c -o $(P2)

ring4: mpi_ring_v4.c
	$(CC) mpi_ring_v4.c -o ring4

sieve_mpi: sieve_mpi.c
	$(CC) sieve_mpi.c -o sieve_mpi -lm

clean:
	rm -vf $(P2) ring4 sieve_mpi
