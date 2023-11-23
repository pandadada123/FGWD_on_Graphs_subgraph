# Subgraph Matching via Fused Gromov-Wasserstein Distance
master thesis 

Libraries:
lib0: Implementation by definition (simple but computationally expensive)
lib1: Implementation with Peyre's trick (mostly use in the thesis)
lib2: Slight modification of lib1, specifically for KEGG database

Main results:
subgraph_LargestValue: A simple example of the largest possible values of WD, GWD, and FGWD for subgraph matching

Performance evaluation:
Synthetic datasets: datasets contains 1000 pairs of randomly created query graph and test graphs
subgraph_build_random_og: with SOT
subgraph_build_random_sliding: with SSOT
subgraph_nema_random: search with NeMa 

Real-world datasets: 
subgraph_dataset_og: search within a large single graph with SOT
subgraph_dataset_sliding: search within a large single graph with SSOT
subgraph_dataset2: search within multiple graphs (search each query in every test graphs) with SSOT
subgraph_dataset2.1: search within multiple graphs (search each query in the current test graph) with SSOT
subgraph_nema_dataset: search within multiple graphs (search each query in the current test graph) with NeMa

Applications:
subgraph_dataset_og: frequent subgraph matching on BZR dataset with SOT
subgraph_kegg_og: on KEGG database with SOT
subgraph_kegg_sliding: on KEGG database with SSOT, with special focus on top-k matching

Future directions:
subgraph_DiffSize_og: A simple example of finding a subgraph of a different size of the query graph with SOT
subgraph_DiffSize_sliding: A simple example of finding a subgraph of a different size of the query graph with SSOT
