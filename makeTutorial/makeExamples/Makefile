objects = hello.o factorial.o main.o
edit : hello
	g++ -o edit $(objects)
hello : $(objects)
	g++ hello.o factorial.o main.o -o hello
hello.o : hello.cpp functions.h
	g++ -c hello.cpp
factorial.o : factorial.cpp functions.h
	g++ -c factorial.cpp
main.o : main.cpp functions.h
	g++ -c main.cpp
clean:
	rm edit hello $(objects)
