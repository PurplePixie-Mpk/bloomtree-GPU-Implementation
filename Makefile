CXX=nvcc
FLAGS=-std=c++11
all: bfs color scc
bfs:bfs.cu
	$(CXX) $^ $(FLAGS) -o $@ 
color:color.cu
	$(CXX) $^ $(FLAGS) -o $@ 
scc:scc.cu
	$(CXX) $^ $(FLAGS) -o $@ 
test:testNeighbours.cu
	$(CXX) $^ $(FLAGS) -o $@ 