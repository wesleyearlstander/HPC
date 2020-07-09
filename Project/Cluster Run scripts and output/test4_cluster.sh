for i in 1 2 3 4 5 6 7 8 9 10; do
	for epoch in 1 3 5 7 9 11 13 15 17 19 21; do 
		mpiexec -n 4 ./main2 5 "${epoch}"
	done
	for layer in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do 
		mpiexec -n 4 ./main2 "${layer}" 5
	done
done
