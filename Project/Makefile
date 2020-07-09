NVCC = nvcc
OBJFILE1 = kernel.cu

TARGET1 = proj


all:   proj



proj: kernel.cu
	nvcc -arch=sm_35 -rdc=true kernel.cu -o proj -lcudadevrt



clean:
	rm -f $(TARGET0) *~
