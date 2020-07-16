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

/*
Required variables for bloomtree:
num_vertices
m_num_hashes //for bloom filter
m_bits //our bloom filter -> in GPU => allocate using cudaHostAlloc in main? or just cudaMalloc?
filter_size //size of our bloom filter
*/

__device__ void Mhash(const long data, uint64_t filter_size, uint64_t & hash1, uint64_t & hash2) 
{
	hash1 = data % filter_size;
	hash2 = (data*data) % filter_size;
}

__device__ inline uint64_t NthHash(uint8_t n, uint64_t hashA, uint64_t hashB, uint64_t filter_size) 
{
	return (hashA + n * hashB) % filter_size;
}

__device__ int LeftChild(int node) 
{
	return (((node + 1) << 1) - 1);
}

__device__ int RightChild(int node) 
{
	return ((node + 1) << 1);
}

__device__ int Sibling(int node) 
{
	return (((node + 1) ^ 1) - 1);
}

__device__ int Parent(int node) 
{
	return (((node + 1) >> 1) - 1); 
}

__device__ void SetBloom(bool *m_bits, long data)
{
	uint64_t hash_values[2];
	Mhash(data, filter_size, hash_values[0], hash_values[1]);
	for (int n = 0; n < m_num_hashes; n++) 
	{
		m_bits[NthHash(n, hash_values[0], hash_values[1], filter_size)] = true;
	}
}

__device__ bool CheckBloom(bool *m_bits, long data)
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

__device__ void Bset(bool *m_bits, l src, l dest, bool path[], l dir_change_idx) 
{
	l cur = Parent(src + num_vertices - 1);
	l i = 1;
	bool dir = true;  // true in upward direction.
    while(cur != (dest + num_vertices - 1))
    {
        if (dir && i != dir_change_idx) 
        {
			SetBloom(m_bits, ((long)cur * num_vertices + src) << 1);
			cur = Parent(cur);
		}
        else if (dir)
        {
			SetBloom(m_bits, (((long)cur * num_vertices + src) << 1) + 1);
			dir = false;
			if (path[i] == 0) cur = LeftChild(cur);
			else cur = RightChild(cur);
		}
        else 
        {
            if (path[i] == 0) 
            {
				SetBloom(m_bits, ((long)cur * num_vertices + src) << 1);
				cur = LeftChild(cur);
			}
            else 
            {
				SetBloom(m_bits, (((long)cur * num_vertices + src) << 1) + 1);
				cur = RightChild(cur);
			}
		}
		i++;
	}
}

__device__ void CopyReverseArray(int len, bool ret[], bool path[])
{
	l id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<len)
	{
		path[id]=ret[len-id-1];
	}
}

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
	// Both the for loops have dependencies, hence can't parallelize
}

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
__global__ void AddEdge(bool *m_bits, l u, l v) 
{
	bool path[2*MAX_HEIGHT];
	int dir_change_idx;
	l len = GetEdgePath(u, v, path, dir_change_idx);
	Bset(m_bits, u, v, path, dir_change_idx);
	for (int i = 0; i < len/2; i++) 
	{
		bool temp = path[len-i-1];
		path[len-i-1] = path[i];
		path[i] = temp;
	}
	// ReverseArray<<<1,len/2>>>(len, path);
	dir_change_idx = len - dir_change_idx;
	Bset(m_bits, v, u, path, dir_change_idx);
}

__device__ bool IsEdge(bool *m_bits, l u, l v) 
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
			if (CheckBloom(m_bits, ((long)cur * num_vertices + u) << 1) == 0)
			{
				return false;
			} 
			cur = Parent(cur);
		}
        else if (dir)
        {
			if (CheckBloom(m_bits, (((long)cur * num_vertices + u) << 1) + 1) == 0)
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
				if (CheckBloom(m_bits, ((long)cur * num_vertices + u) << 1) == 0)
				{
					return false;
				} 
				cur = LeftChild(cur);
			}
            else 
            {
				if (CheckBloom(m_bits, (((long)cur * num_vertices + u) << 1) + 1) == 0)
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

__global__ void Neighbours(bool *m_bits, int u, bool *neigh)  //length of neigh is num_vertices.
{
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<num_vertices)
	{
		neigh[id] = IsEdge(m_bits, u, id);
	}
}