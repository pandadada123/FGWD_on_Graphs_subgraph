# Subgraph Matching via Fused Gromov-Wasserstein Distance
MSc Thesis link here: http://resolver.tudelft.nl/uuid:d18d7083-4189-46a7-bd03-4cf49674cffe

#### Libraries
- [lib0](/lib0): Implementation by definition (simple but computationally expensive)
- [lib1](/lib1): Implementation with Peyre's trick (mostly used in the thesis)
- [lib2](/lib2): Slight modification of lib1, specifically for KEGG database

#### Normalized Fused Gromov-Wasserstein (nFGW) Distance
- [subgraph_LargestValue](/subgraph_LargestValue): A simple example of the largest possible values (approximately 1.0) of nWD, nGWD, and nFGWD for subgraph matching

#### Also check [FGWD_on_Graphs_basics](https://github.com/pandadada123/FGWD_on_Graphs_basics) for visualization of GW and FGW nonconvexity 

## Main results 
### Performance evaluation
#### Synthetic datasets: datasets contains 1000 pairs of randomly created query graph and test graphs
- [subgraph_build_random_og](/subgraph_build_random_og): with SOT
- [subgraph_build_random_sliding](/subgraph_build_random_sliding): with SSOT
- [subgraph_nema_random](/subgraph_nema_random): search with NeMa 

#### Real-world datasets
- [subgraph_dataset_og](/subgraph_dataset_og): search within a large single graph with SOT
- [subgraph_dataset_sliding](/subgraph_dataset_sliding): search within a large single graph with SSOT
- [subgraph_dataset2](/subgraph_dataset2): search within multiple graphs (search each query in every test graph) with SSOT
- [subgraph_dataset2.1](/subgraph_dataset2.1): search within multiple graphs (search each query in the current test graph) with SSOT
- [subgraph_nema_dataset](/subgraph_nema_dataset): search within multiple graphs (search each query in the current test graph) with NeMa

For G-Finder, two files are modified within the original project 
- [GFinder_modified](/GFinder_modified)

### Applications
- [subgraph_dataset_og](/subgraph_dataset_og): frequent subgraph matching on BZR dataset with SOT
- [subgraph_kegg_og](/subgraph_dataset_og): on KEGG database with SOT
- [subgraph_kegg_sliding](/subgraph_kegg_sliding): on KEGG database with SSOT, with a special focus on top-k matching

### Future directions
Searching for subgraphs of a different size of the query
- [subgraph_DiffSize_og](/subgraph_DiffSize_og): A simple example of finding a subgraph of a different size of the query graph with SOT
- [subgraph_DiffSize_sliding](/subgraph_DiffSize_sliding): A simple example of finding a subgraph of a different size of the query graph with SSOT
