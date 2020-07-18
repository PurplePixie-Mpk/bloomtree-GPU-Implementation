/*--------------------------------------------------------------------
BFS using Bloom Tree. 
Input format: ./bfs filename num_vertices num_bits num_hash_functions 
--------------------------------------------------------------------*/

#include <bits/stdc++.h>
#include "bloomtree.cu"
using namespace std;

#define l int
#define DISPLAY_BFS_DIST true
#define N_THREADS_PER_BLOCK 1024
const l N = 2e5 + 5;
const l INF = 1e8;



queue<int> q;
// Initialises bfs_dist to INF
__global__ void Init(int bfs_dist[N]) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	bfs_dist[i] = INF;
}

/*__global__ void parallelize_level(queue<int> q,int adj[N],int u,int bfs_dist[N])
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if(bfs_dist[adj[tid]] == INT_MAX)
	{
		bfs_dist[adj[tid]] = bfs_dist[u] + 1;
		q.push(adj[tid]);
	}
}*/

/*__global__ void print(int bfs_dist[N])
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (bfs_dist[tid] != INF)
	{
		cout << tid << " - " << bfs_dist[tid] << "\n";
	}
}*/

void BreadthFirstSearch(int s) {
	l bfs_dist[N],*device_bfs_dist;
	cudaMemcpy(device_bfs_dist, bfs_dist, sizeof(int) * N, cudaMemcpyHostToDevice);
	int n_blocks = (num_vertices+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK;
	Init<<<n_blocks,N_THREADS_PER_BLOCK>>>(device_bfs_dist);
	cudaMemcpy(bfs_dist, device_bfs_dist, sizeof(int) * N, cudaMemcpyDeviceToHost);
	q.push(s);
	bfs_dist[s] = 0;
	//int level = 1;
	
	while(!q.empty()){
		int u = q.front();
		q.pop();

		bool adj[num_vertices];
		GetNeighbours(s,adj);
		int vert_adj[N],j=0;
		for(int i=0;i<num_vertices;i++)
		{
			if(adj[i]==true)
			{
				vert_adj[j]=i;
				j++;
			}
		}
		for (l i = 0; i < j; ++i) {
			if (bfs_dist[vert_adj[i]] == INF){
				bfs_dist[vert_adj[i]] = bfs_dist[u] + 1;
				q.push(vert_adj[i]);
			}
		}
	}

	if (DISPLAY_BFS_DIST) {
		for (int i = 0; i < num_vertices; ++i) {
			if (bfs_dist[i] != INF) {
				cout << i << " - " << bfs_dist[i] << "\n";
			}
		}
	}

}

int main(int argc,char** argv){
	if (argc != 5) {
		cout << "Input format: ./program_name filename num_vertices num_bits num_hash_functions\n";
		return 0;
	}
	
	clock_t ti;
	
	num_vertices = atoi(argv[2]);
	int num_bits = atoi(argv[3]);
	int num_hash_funs = atoi(argv[4]);
	
	InitBloomTree(num_vertices,num_bits,num_hash_funs);

	ifstream fin(argv[1], ios::in);
	while(!fin.eof()){
		l u,v;
		fin >> u >> v;
		if (u == v) continue;
		AddEdge(u,v);
	}
	fin.close();

	ti = clock();
	BreadthFirstSearch(0);
	ti = clock() - ti;
	
	printf("%.5f\n", float(ti) / CLOCKS_PER_SEC);
	return 0;
}