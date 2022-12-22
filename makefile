LOU_cu := LouvainPrllel.cu ./source/Helper.cu ./source/Phase1.cu ./source/Phase2.cu
LOU_cpp := LouvainSer.cpp ./source/Louvain.cpp

LouvainCPP : $(LOU_cpp)
	g++ $(LOU_cpp) -o run
	./run

LouvainCU : $(LOU_cu)
	nvcc $(LOU_cu) -o run
	./run

clear :
	rm run
