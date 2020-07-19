/****************************************************************************************/
/* 	CS023 Course Project: Bloomtree GPU implementation using CUDA library				*/
/* 	CUDA implementation of all relevant functions										*/
/* 																						*/
/****************************************************************************************/

#include<bits/stdc++.h>
#include<cuda.h>
using namespace std;

#define l int
#define pb push_back
#define rep(i, a, b) for (l i = a; i < b; ++i)

#define MAX_SIZE 1e8
#define MAX_HEIGHT 25 //Maximum number of nodes = around 33.5 million

__managed__ int num_vertices;
__managed__ uint8_t m_num_hashes;
__managed__ uint64_t filter_size;
__managed__ bool *m_bits;

/*
Required variables for bloomtree:
num_vertices
m_num_hashes //for bloom filter
m_bits //our bloom filter -> in GPU => allocate using cudaHostAlloc in main? or just cudaMalloc?
filter_size //size of our bloom filter
*/

// Helper function to calculate two hash functions which is later used to calculate the Nth hash function
__device__ void Mhash(const long data, uint64_t filter_size, uint64_t & hash1, uint64_t & hash2) 
{
	hash1 = data;
	hash2 = (data*data);
}

// Helper function to calculate Nth hash function
__device__ inline uint64_t NthHash(uint8_t n, uint64_t hashA, uint64_t hashB, uint64_t filter_size) 
{
	return (hashA + n * hashB) % filter_size;
}


// To get index of leftChild of a given node
__device__ int LeftChild(int node) 
{
	return (((node + 1) << 1) - 1);
}

// To get index of rightChild of a given node
__device__ int RightChild(int node) 
{
	return ((node + 1) << 1);
}

// To get index of the sibling of a given node
__device__ int Sibling(int node) 
{
	return (((node + 1) ^ 1) - 1);
}

// To get index of parent of a given node
__device__ int Parent(int node) 
{
	return (((node + 1) >> 1) - 1); 
}

// Utility function to set bloomfilter as per the given data
__device__ void SetBloom(long data)
{
	uint64_t hash_values[2];
	Mhash(data, filter_size, hash_values[0], hash_values[1]);
	for (int n = 0; n < m_num_hashes; n++) 
	{
		m_bits[NthHash(n, hash_values[0], hash_values[1], filter_size)] = true;
	}
}

// Utility function to check if bloomfilter as per the given data is set or not
__device__ bool CheckBloom(long data)
{
	uint64_t hash_values[2];
	Mhash(data, filter_size, hash_values[0], hash_values[1]);
	for (int n = 0; n < m_num_hashes; n++) 
	{
		if (!m_bits[NthHash(n, hash_values[0], hash_values[1], filter_size)]) 
		{
			return false;
		}
	}
	return true;
}

// Utility function to set all bits corresponding to a given path in the bloom tree
__device__ void Bset(l src, l dest, bool path[], l dir_change_idx) 
{
	l cur = Parent(src + num_vertices - 1);
	l i = 1;
	bool dir = true;  // true in upward direction.
    while(cur != (dest + num_vertices - 1))
    {
        if (dir && i != dir_change_idx) 
        {
			SetBloom(((long)cur * num_vertices + src) << 1);
			cur = Parent(cur);
		}
        else if (dir)
        {
			SetBloom((((long)cur * num_vertices + src) << 1) + 1);
			dir = false;
			if (path[i] == 0) cur = LeftChild(cur);
			else cur = RightChild(cur);
		}
        else 
        {
            if (path[i] == 0) 
            {
				SetBloom(((long)cur * num_vertices + src) << 1);
				cur = LeftChild(cur);
			}
            else 
            {
				SetBloom((((long)cur * num_vertices + src) << 1) + 1);
				cur = RightChild(cur);
			}
		}
		i++;
	}
}

// Function to copy reversed array in parallel
__device__ void CopyReverseArray(int len, bool ret[], bool path[])
{
	l id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<len)
	{
		path[id]=ret[len-id-1];
	}
}

// Function to get path of a node in bloom tree from the root
__device__ int GetVertexPath(l node, bool path[]) 
{
	node = node + num_vertices - 1;
	bool ret[MAX_HEIGHT];
	int len = 0;
    while (node != 0) 
    {
		l par = Parent(node);
		if (node == LeftChild(par))
		{
			ret[len]=0;
			len++;
		}
		else
		{
			ret[len]=1;
			len++;
		}
		node = par;
	}
	for (int i = 0; i < len; i++) 
	{
		path[i] = ret[len-i-1];
	}
	// CopyReverseArray<<<1,len>>>(len, ret, path);
	return len;
}

// Function to get path of a node from another node in the bloom tree
__device__ int GetEdgePath(l u, l v, bool path[], l& dir_change_idx) 
{
	bool path_u[MAX_HEIGHT], path_v[MAX_HEIGHT];
	l len_u = GetVertexPath(u, path_u);
	l len_v = GetVertexPath(v, path_v);
	l i = 0;
	for(; i < len_u && i < len_v && path_u[i] == path_v[i]; ++i);
	l idx = 0;
	rep(j, 0, len_u - i)
	{
		path[idx] = (path_u[len_u - j - 1]);
		idx++;
	}
	dir_change_idx = idx;
	rep(j, i, len_v)
	{
		path[idx] = (path_v[j]);
		idx++;
	}
	return idx;
	// Both the for loops have dependencies, hence can't parallelize these loops
}

// Function to reverse an array parallely
__device__ void ReverseArray(int len, bool path[])
{
	l id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<len/2)
	{
		bool temp = path[len-id-1];
		path[len-id-1] = path[id];
		path[id] = temp;
	}
}

//use one thread to do this:
__global__ void AddEdgeBloomTree(l u, l v) 
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id!=0)
		return;
	bool path[2*MAX_HEIGHT];
	int dir_change_idx;
	l len = GetEdgePath(u, v, path, dir_change_idx);
	Bset(u, v, path, dir_change_idx);
	for (int i = 0; i < len/2; i++) 
	{
		bool temp = path[len-i-1];
		path[len-i-1] = path[i];
		path[i] = temp;
	}
	// ReverseArray<<<1,len/2>>>(len, path);
	dir_change_idx = len - dir_change_idx;
	Bset(v, u, path, dir_change_idx);
}

__device__ bool IsEdge(l u, l v) 
{
	if(u==v)
		return false;
	bool path[2*MAX_HEIGHT];
	int dir_change_idx;
	GetEdgePath(u, v, path, dir_change_idx);
	l cur = Parent(u + num_vertices - 1), i = 1;
	bool dir = true;
    while (cur != (v + num_vertices - 1))
    {
        if (dir && i != dir_change_idx) 
        {
			if (CheckBloom(((long)cur * num_vertices + u) << 1) == 0)
			{
				return false;
			} 
			cur = Parent(cur);
		}
        else if (dir)
        {
			if (CheckBloom((((long)cur * num_vertices + u) << 1) + 1) == 0)
			{
				return false;
			} 
			dir = false;
			if (path[i] == 0) cur = LeftChild(cur);
			else cur = RightChild(cur);
		}
        else 
        {
            if (path[i] == 0) 
            {
				if (CheckBloom(((long)cur * num_vertices + u) << 1) == 0)
				{
					return false;
				} 
				cur = LeftChild(cur);
			}
            else 
            {
				if (CheckBloom((((long)cur * num_vertices + u) << 1) + 1) == 0)
				{
					return false;
				} 
				cur = RightChild(cur);
			}
		}
		++i;
	}
	return true;
}

// Function to get neighbours of a specified node in the original graph
__global__ void Neighbours(int u, bool *neigh)  //length of neigh is num_vertices.
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<num_vertices)
	{
		neigh[id] = IsEdge(u, id);
	}
}

// Helper function to initialize all elements to false
__global__ void InitAllToFalse(bool *array, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size)
	{
		array[tid]=false;
	}
}

// Function to initialize the bloomTree
void InitBloomTree(l numVertices, l filterSize, l numHashes)
{
	num_vertices = numVertices;
	filter_size = filterSize;
	m_num_hashes = numHashes;
	cudaMalloc(&m_bits, filterSize * sizeof(bool));
	InitAllToFalse<<<(filterSize/1024 + 1), 1024>>>(m_bits, filterSize);
}

// Function to be called from CPU to add an edge
void AddEdge(int u, int v)
{
	AddEdgeBloomTree<<<1,1>>>(u,v);
}

// Function to be called from CPU to get neighbours of a particular vertex
void GetNeighbours(int u, bool *neigh)
{
	Neighbours<<<(num_vertices/1024 + 1),1024>>>(u, neigh);
}
