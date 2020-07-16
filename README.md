# bloomtree
CUDA Implementation of BloomTree - A Space-Efficient Approximate Representation for Graphs

## BloomTree class
```
#include "bloomtree.cu"
```

Initialise the bloom filter used for the bloom tree with the size of bloom filter, and number of hash functions
```
InitBloomTree(int numVertices, int filterSize, int numHashes);
```

Methods provided by our bloomtree implementation are
```
AddEdge(int u, int v)
GetNeighbours(int u, bool[] neighbours)
```

## Test
```
$ make test
$ ./test <path_to_graph> num_vertices num_bits num_hash_functions
```

## Run
```
$ make all
$ ./bfs <path_to_graph> num_vertices num_bits num_hash_functions
$ ./color <path_to_graph> num_vertices num_bits num_hash_functions
$ ./scc <path_to_graph> num_vertices num_bits num_hash_functions
```

## References
* https://github.com/PurplePixie-Mpk/bloomtree
* http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/