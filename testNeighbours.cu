/****************************************************************************************/
/* 	CS023 Course Project: Bloomtree in GPU using CUDA library							*/
/* 	Testing bloomtree implementation using Neighbours function							*/
/* 																						*/
/****************************************************************************************/

#include <bits/stdc++.h>
#include "bloomtree.cu"
#include <cuda.h>

using namespace std;

#define rep(i, a, b) for (l i = a; i < b; ++i)

__global__ void InitAllToFalse(bool *array, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size)
	{
		array[tid]=false;
	}
}

int main(int argc,char** argv)
{
	if (argc != 5) 
	{
		cout << "Input format: ./program_name input_filename num_vertices num_bits num_hash_functions\n";
		return 0;
	}

	// Parsing command line arguments
	num_vertices = atoi(argv[2]);
	filter_size = atoi(argv[3]);
	m_num_hashes = atoi(argv[4]);

	// Initializing bloom_filter for our bloom tree
	bool *m_bits;
	cudaMalloc(&m_bits, filter_size * sizeof(bool));
	InitAllToFalse<<<(filter_size/1024 + 1), 1024>>>(m_bits, filter_size);

	// Reading input and adding edges
	ifstream fin(argv[1], ios::in);
	while(!fin.eof())
	{
		l u,v;
		fin >> u >> v;
		if (u == v) continue;
		AddEdge<<<1,1>>>(m_bits,u,v);
	}
	fin.close();

	// Testing Neighbours function
	bool *neigh, *neighCpu;
	cudaMalloc(&neigh, num_vertices * sizeof(bool));
	InitAllToFalse<<<(num_vertices/1024 + 1), 1024>>>(neigh, num_vertices);
	Neighbours<<<(num_vertices/1024 + 1),1024>>>(m_bits, 1, neigh);

	// Copying output in GPU back to CPU and displaying the result
	neighCpu = (bool *)malloc(num_vertices * sizeof(bool));
	cudaMemcpy(neighCpu, neigh, num_vertices * sizeof(bool), cudaMemcpyDeviceToHost);
	rep(ii,0,num_vertices) 
	{
		cout<<neighCpu[ii]<<' ';
	}
	return 0;
}