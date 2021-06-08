#include <bits/stdc++.h>
#include "bloomtree.cu"
using namespace std;

#define l int
#define DISPLAY_COLORS true
const l N = 9;
#define N_THREADS_PER_BLOCK 1024

int adj_size;

__global__ void assign(bool is_neighbour_colour[N],int color[N])
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	is_neighbour_colour[tid] = false; 
	color[tid]=-1;
}

__global__ void check(bool is_neighbour_color[N],int color[N],int adj[N])
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;;
	if (color[adj[i]] != -1)
		is_neighbour_color[color[adj[i]]] = true;
}

void Colouring(){
	int *color = new l[num_vertices];
	bool *is_neighbour_color = new bool[num_vertices+1];
	
	// Initially, there is no color to any neighbour and to any vertex
	//int n_blocks = (num_vertices+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK;

	
	for(int v=1;v<=num_vertices;v++)
	{
		color[v]=-1;
		is_neighbour_color[v]=false;
	}
	color[0] = 1;
	int num_colors_used = 1;

	bool *t_adj,*t_gpu_adj;
	cudaMalloc(&t_adj, num_vertices * sizeof(bool));
	t_gpu_adj = (bool *)malloc(num_vertices * sizeof(bool));
	
	
	for (int v = 1; v < num_vertices; ++v)
	{
		//cout<<v<<"\n";
		InitAllToFalse<<<(num_vertices/1024 + 1), 1024>>>(t_adj, num_vertices);
		GetNeighbours(v, t_adj);
		cudaMemcpy(t_gpu_adj, t_adj, num_vertices * sizeof(bool), cudaMemcpyDeviceToHost);
		vector<int> adj;
		//InitAllToFalse<<<(num_vertices/1024 + 1), 1024>>>(is_neighbour_color, num_vertices+1);
		for(int i=0;i<num_vertices;i++)
		{
			if(t_gpu_adj[i]==true)
			{
				//cout<<i<<" ADJ "<<v<<"\n";
				adj.push_back(i);
				if(color[i]!=-1)
					is_neighbour_color[color[i]]=true;
				
			}
		}
		//for(int i=1;i<=num_vertices;i++)
		//	cout<<is_neighbour_color[i]<<"C\n";
		

		// Finding first unassigned colour
		l c;
		for (c = 1; c <= num_vertices; ++c) {
			if (is_neighbour_color[c] == false) break;
		}
		color[v] = c;
		for (l i = 0; i < adj.size(); ++i) {
			//cout<<i<<" "<<color[adj[i]]<<"\n";
			if (color[adj[i]] != -1)
				is_neighbour_color[color[adj[i]]] = false;
		}

		if(c > num_colors_used) num_colors_used = c;
	}
	
	if (DISPLAY_COLORS) {
		for(l v = 0; v < num_vertices; ++v) {
			cout << v << " - " << color[v] << "\n";
		}
	}

	cout << "Number of colours used: " << num_colors_used << "\n";
}
int main(int argc, char** argv){
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
	Colouring();
	ti = clock() - ti;
	
	printf("%.5f\n", float(ti) / CLOCKS_PER_SEC);
	return 0;
}