#include <bits/stdc++.h>
#include "bloomtree.cu"
using namespace std;

#define l int
#define DISPLAY_SCC false
#define N_THREADS_PER_BLOCK 1024
const int N = 2e5 + 5;
const int INF = 1e8;
int WH = -1, BL = 1, num_sccs;
int dfs_num[N], dfs_low[N], dfscounter;
stack<int> dfs_scc;
set<int> in_stack;

__global__ void init(int dfs_num[N]){
	int i = blockDim.x*blockIdx.x + blockIdx.x;
		dfs_num[i]=-1;
}

void Init(){
	int n_blocks = (num_vertices+N_THREADS_PER_BLOCK-1)/N_THREADS_PER_BLOCK-1,*d_dfs_num;
	cudaMemcpy(d_dfs_num, dfs_num, sizeof(int) * N, cudaMemcpyHostToDevice);
	init<<<n_blocks,N_THREADS_PER_BLOCK>>>(d_dfs_num);
	cudaMemcpy(dfs_num, d_dfs_num, sizeof(int) * N, cudaMemcpyDeviceToHost);
}

void SCC(int u) {
	dfs_low[u] = dfs_num[u] = dfscounter++;
	dfs_scc.push(u);
	in_stack.insert(u);
	
	bool adja[num_vertices];
	GetNeighbours(u, adja);
	vector<int> adj;
	for(int i=0;i<num_vertices;i++)
	{
		if(adja[i]==true)
			adj.push_back(i);
	}
	
	for (int i = 0; i < adj.size(); ++i) {	
		if (dfs_num[adj[i]] == WH) {
			SCC(adj[i]);
		}
		if (in_stack.find(adj[i]) != in_stack.end()) {
			dfs_low[u] = min(dfs_low[u], dfs_low[adj[i]]);
		}
	}
	
	if (dfs_low[u] == dfs_num[u]) {  // u is a root of a SCC.
		num_sccs++;
		while (!dfs_scc.empty() && dfs_scc.top() != u){
			if (DISPLAY_SCC) {
				cout << dfs_scc.top() << " ";
			}
			in_stack.erase(dfs_scc.top());
			dfs_scc.pop();
		}
		if (DISPLAY_SCC) {
			cout << dfs_scc.top() << "\n";
		}
		in_stack.erase(dfs_scc.top());
		dfs_scc.pop();
	}
}
int main(int argc, char** argv) {
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
	Init();
	dfscounter = 0;
	num_sccs = 0;
	for (int i = 0; i < num_vertices; ++i) {
		if (dfs_num[i] == WH) {
			SCC(i);
		}
	}
	ti = clock() - ti;
	
	printf("%.5f\n", float(ti) / CLOCKS_PER_SEC);
	cout << "Number of SCCs: " << num_sccs << "\n";
	return 0;
}