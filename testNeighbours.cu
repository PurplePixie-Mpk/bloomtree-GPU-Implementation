/****************************************************************************************/
/* 	CS023 Course Project: Bloomtree GPU implementation using CUDA library				*/
/* 	Testing bloomtree implementation using Neighbours function							*/
/* 	Input format: ./test filename num_vertices num_bits num_hash_functions				*/
/****************************************************************************************/

#include <bits/stdc++.h>
#include "bloomtree.cu"
#include <cuda.h>

using namespace std;

#define rep(i, a, b) for (l i = a; i < b; ++i)

int main(int argc,char** argv)
{
	if (argc != 5) 
	{
		cout << "Input format: ./program_name input_filename num_vertices num_bits num_hash_functions\n";
		return 0;
	}

	// Parsing command line arguments
	int numVertices, filterSize, numHashes;
	numVertices = atoi(argv[2]);
	filterSize = atoi(argv[3]);
	numHashes = atoi(argv[4]);

	// Initializing bloom_filter for our bloom tree
	InitBloomTree(numVertices, filterSize, numHashes);

	// Reading input and adding edges
	ifstream fin(argv[1], ios::in);
	while(!fin.eof())
	{
		l u,v;
		fin >> u >> v;
		if (u == v) continue;
		AddEdge(u,v);
	}
	fin.close();

	// Testing Neighbours function
	bool *neigh, *neighCpu;
	cudaMalloc(&neigh, numVertices * sizeof(bool));
	InitAllToFalse<<<(numVertices/1024 + 1), 1024>>>(neigh, numVertices);
	GetNeighbours(1, neigh);

	// Copying output in GPU back to CPU and displaying the result
	neighCpu = (bool *)malloc(numVertices * sizeof(bool));
	cudaMemcpy(neighCpu, neigh, numVertices * sizeof(bool), cudaMemcpyDeviceToHost);
	rep(ii,0,numVertices) 
	{
		cout<<neighCpu[ii]<<' ';
	}
	return 0;
}