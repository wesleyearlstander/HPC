Type Make to compile
Make clean to removed compilation

Run program as mpiexec -n {processes} ./main2 {layers} {max_epoch}

Run program as mpiexec -n 2 ./main2 3 1000 to get best results

Speedup will be in output.txt or printed in the console window

To run approach one allocated 14 processes, to run approach two allocate less than 14 processes.

Max epoch is set to 1 so increase in parameters to see accuracy increase.

Run scripts for approach one and approach two are added.
