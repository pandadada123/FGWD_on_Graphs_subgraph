// FastDynamic.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>
#include <ostream>
#include <iostream>
#include <cmath>
#include <time.h>
#include "transform_query_file.h"
#include "pre_process_data_graph.h"
#include "time.h"
#include "utility.h"
#include "cfl_decomposition.h"
#include "seed_finder.h"
#include "cpi_builder.h"
#include "matching_order.h"
#include "enumeration.h"
#include "sim_base_build_cpi.h"
#include "core_query_tree_build.h"
#include "inexact_matching_order.h"
#include "find_result.h"
#include <vector>
#include <utility>
#include <tuple>

// #include <filesystem>


CUtility * cu_total = new CUtility();
CUtility *cu_readQuery = new CUtility();
CUtility *cu_prepare = new CUtility();
CUtility *cu_simulation = new CUtility();
CUtility *cu_getSequence = new CUtility();
CUtility *cu_querying = new CUtility();

#define TOTAL_BEGIN  cu_total->startCT();
#define TOTAL_END g_time_total = cu_total->endCT(false);

#define READ_QUERY_BEGIN cu_readQuery->startCT();
#define READ_QUERY_END g_time_readQuery = cu_readQuery->endCT(false);

#define PREPARE_BEGIN cu_prepare->startCT();
#define PREPARE_END g_time_prepare = cu_prepare->endCT(false);

#define SIMULATION_BEGIN cu_simulation->startCT();
#define SIMULATION_END g_time_simulation = cu_simulation->endCT(false);

#define GET_SEQ_BEGIN  cu_getSequence->startCT();
#define GET_SEQ_END g_time_getSequence = cu_getSequence->endCT(false);

#define SEARCH_BEGIN cu_querying->startCT();
#define SEARCH_END g_time_search = cu_querying->endCT(false);


void getLimit_full(string str_full_limit, long & LIMIT) {
	if (str_full_limit == "1K")
		LIMIT = 1000;
	else if (str_full_limit == "10K")
		LIMIT = 10000;
	else if (str_full_limit == "100K")
		LIMIT = 100000;
	else if (str_full_limit == "100M")
		LIMIT = 100000000;
	else if (str_full_limit == "1B")
		LIMIT = 100000000000;
	else
		LIMIT = atol(str_full_limit.c_str());
}


void printVectorInt(vector<int> &vet) {
	for (vector<int>::iterator pos = vet.begin(); pos != vet.end(); pos++) {
		printf("%d ", *pos);
	}
	putchar('\n');
}

void myy(vector<int> &a) {
	for (int i = 0; i < a.size(); i++) {
		cout << i;
	}
}


inline bool is_contain_empty_indexset() {
	for (int step = 0; step < g_core_size; step++) {
		CPINode* tmp_node = &indexSet[step];
		if (tmp_node->size == 1 && tmp_node->candidates[0] == -1) {
			return true;
		}
	}
	return false;
}

inline void resetGlobalVariables_1() {
	// Cleaning up at the end of the function:
	g_need_clean.clear();
	g_forward_build_sequence.clear();
	memset(g_forward_build_parent, -9999, sizeof(long long) * MAX_QUERY_NODE);
	memset(g_forward_level, 0, sizeof(int) * MAX_QUERY_NODE);
	for (int i = 1; i <= g_level_size; ++i) {
		g_level[i].clear();
	}
	memset(g_visited_for_query_graph, 0, sizeof(char) * g_cnt_node_query_graph);
	g_core_tree_node_child_array_index = 0;
	g_core_tree_node_nte_array_index = 0;
	g_core_size = 1;
	g_level_size = 0;
}

inline void resetGlobalVariables_2() {
	// Reinitialize the variables
	memset(g_visited_for_query_graph, 0, sizeof(g_visited_for_query_graph));  // Assuming char array
	g_cnt_node_query_graph = 0;
	g_nodes_adj_list_with_edge_type_query_graph.clear();
	g_is_init_edge_matchinig_array = false;
	g_level_size = 0;
	//g_level.clear();
	memset(g_nodes_label_query_graph, 0, sizeof(g_nodes_label_query_graph)); // Assuming int array
	g_cnt_unique_label_data_graph = 0;
	g_good_count_need_clean_size = 0;
	
	g_one_hop_label_count_query_graph = NULL;
	g_two_hop_label_count_query_graph = NULL;

	// Free dynamically allocated memory and reinitialize pointers to NULL if needed
	// (Do this only if you're sure no other part of the program will access them after this function)
	// For example:
	// delete[] somePointerVariable;
	// somePointerVariable = NULL;
}


// int main(int argc, char *argv[])
int main()
{

	// use for directed graph
	std::string mean_fea = "0"; // number of nodes that has been changed
	std::string std_fea = "0"; // zero mean Gaussian
	//std::string str_mean = "0";
	//std::string str_std = "0.1";

	//std::string dataset_n = "bzr";
	std::string dataset_n = "firstmm";

	std::string targetFolderPath = "E:\\Master Thesis\\comparisons\\GFinder\\test_dataset\\" + dataset_n + "\\target_noise_fea_"+ mean_fea+ "_"+ std_fea + "\\";
	std::string queryFolderPath = "E:\\Master Thesis\\comparisons\\GFinder\\test_dataset\\" + dataset_n + "\\query_noise_fea_" + mean_fea + "_" + std_fea + "\\";

	// const int numFiles = 405;

	// char** argv = new char* [numFiles * 2];  // Allocate memory for argv


	for (int i = 30; i < 41; ++i) {
		for (int j = 0; j < 10; ++j) {
			// Formulate the file paths based on the iteration index

			// Allocate memory for char* and copy the contents
			//argv[i * 2] = new char[std::strlen(targetFolderPath) + std::to_string(i).length() + 8]; // For ".format" and null terminator
			//std::sprintf(argv[i * 2], "%s%d.format", targetFolderPath, i);

			//argv[i * 2 + 1] = new char[std::strlen(queryFolderPath) + std::to_string(i).length() + 8]; // For ".format" and null terminator
			//std::sprintf(argv[i * 2 + 1], "%s%d.format", queryFolderPath, i);

			// Convert int to string
			std::string strNumber = std::to_string(i);
			std::string strNumber2 = std::to_string(j);

			std::string Arg1 = { targetFolderPath + strNumber + ".format" };
			std::string Arg2 = { queryFolderPath + strNumber + "_"+strNumber2+".format" };
			// std::string Arg1 = { "E:\\Master Thesis\\comparisons\\GFinder_dataset\\GFinder_dataset\\nell\\NELL_gfinder_full_7_union_prune.format" };
			// std::string Arg2 = { "E:\\Master Thesis\\comparisons\\GFinder_dataset\\GFinder_dataset\\nell\\NELL_queries7_union.format" };

			// argv[1] = "C:\\Users\\lihui\\workspace\\github\\test_dataset\\data.format";
			// argv[2] = "C:\\Users\\lihui\\workspace\\github\\test_dataset\\query.format";
			// argv[1] = "E:\\Master Thesis\\comparisons\\GFinder\\gfinder\\GFinder-master\\GFinder-master\\test_dataset\\target_noise_0.01\\2.format";
			// argv[2] = "E:\\Master Thesis\\comparisons\\GFinder\\gfinder\\GFinder-master\\GFinder-master\\test_dataset\\query_noise_0.01\\2.format";
			// argv[1] = "E:\\Master Thesis\\comparisons\\GFinder\\gfinder\\GFinder-master\\GFinder-master\\test_dataset\\data5.format";
			// argv[2] = "E:\\Master Thesis\\comparisons\\GFinder\\gfinder\\GFinder-master\\GFinder-master\\test_dataset\\query5.format";

			int YES = 0;
			double rate = 0.0;
			//std::vector<std::pair<int, int>> pairs;


			////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// argv[3] = "-f";
			// argv[4] = "1";
			// argv[5] = "1";

			// argv[1] = "a";
			// argv[2] = "a";

			// top 10
			TOPK = 1;
			NODE_MISS_COST = 1;
			BRIDGE_COST = 1;
			EDGE_MISSING_COST = 1;
			DYNAMIC = true;
			// IS_ONE_HOP_DATA_GRAPH should be equl to 1 or 0
			IS_ONE_HOP_DATA_GRAPH = 1;
			DISTINCT_LABEL = 0;

			// string data_graph_file = argv[1];
			// string query_graph_file = argv[2];
			string data_graph_file = Arg1;
			string query_graph_file = Arg2;

			//TOPK = stoi(argv[3]);
			//NODE_MISS_COST = stoi(argv[4]);
			//BRIDGE_COST = stoi(argv[5]);
			//EDGE_MISSING_COST = stoi(argv[6]);
			//IS_ONE_HOP_DATA_GRAPH = stoi(argv[7]);


			cout << endl << "*******************************************************************" << endl;
			cout << "Data File :" << Arg1 << endl;
			cout << "Query file:" << Arg2 << endl;
			cout << "Top-K: " << TOPK << endl;
			cout << "Missing Node Cost: " << NODE_MISS_COST << endl;
			cout << "Intermediate Cost: " << BRIDGE_COST << endl;
			cout << "Missing Edge Cost: " << EDGE_MISSING_COST << endl;
			cout << "Is One Hop: " << IS_ONE_HOP_DATA_GRAPH << endl;
			cout << "*******************************************************************" << endl;

			// ======= Begin clock ========
			g_clock = clock();

			//==== FIRST step: read and process the data graph ============================
			cout << "Reading data graph ..." << endl;
			preprocessDataGraph(data_graph_file);
			cout << "processing data graph ..." << endl;
			readAndProcessDataGraph(data_graph_file);
			initNECStructure();
			cout << "OK!" << endl;

			cout << "Building index..." << endl;
			cout << "Begin parametersInitilisingBeforeQuery..." << endl;
			initParametersBeforeQuery();
			cout << "Begin initializeStatisticParameters..." << endl;
			initializeStatisticParameters();
			cout << "OK!" << endl;
			//=======FIRST step END =====================================================


			cout << "ATTENTION: Finish data index cost: " << (clock() - g_clock) * 1.0 / CLOCKS_PER_SEC << endl;

			// int count_query_file = atoi(argv[4]);
			int count_query_file = 1;
			//string str_full_limit = argv[5];

			//getLimit_full(str_full_limit, LIMIT);

			char c;
			int query_id;

			for (long long i = 0; i < count_query_file; i++) {
				// clean heap here
				{
					g_result_heap.clear();
					g_maxPartialNum = 0;
					g_mapping_found = 0;
				}

				TOTAL_BEGIN;

				single_readQueryGraph(query_graph_file);

				PREPARE_BEGIN;

				coreDecompositionQuery();
				cout << "Decompose Query OK!" << endl;
				g_isTree = true;
				for (long long i = 0; i < g_cnt_node_query_graph; i++) {
					cout << "g_cnt_node_query_graph" << g_cnt_node_query_graph << endl;   // number of query nodes
					cout << g_core_number_query_graph[i];
					if (g_core_number_query_graph[i] >= 2) {
						g_isTree = false;
						break;
					}
				}
				cout << "g_isTree" << g_isTree << endl;

				g_isTree = false;
				if (g_isTree) {

				}
				else {
					// extract residual tree and NEC from query
					extractResidualStructures();
					g_root_node_id_of_query = selectRootFromQuery();
					//g_root_node_id_of_query = 1;
					cout << "g_root_node_id_of_query:" << g_root_node_id_of_query << endl;
					findRootCandidate();
					int root_index_point = -1;
					cout << "g_root_candidates_size:" << g_root_candidates_size << endl;
					while (g_root_candidates_size == 0) {
						cout << "g_root_candidates_size:" << g_root_candidates_size << endl;
						root_index_point = root_index_point + 1;
						if (root_index_point == query_root_sort_list.size()) {
							cout << "query_root_sort_list.size():" << query_root_sort_list.size() << endl;
							cout << "All node in core has no candidate!" << endl;

							// At the end of the function:
							resetGlobalVariables_1();

							// At the end of the function:
							resetGlobalVariables_2();

							return 0;
						}
						g_root_node_id_of_query = query_root_sort_list[root_index_point] % ONE_M;
						findRootCandidate();
					}

					PREPARE_END;

					for (int region = 0; region < g_root_candidates_size; region++) {

						long long root_cand_id = g_root_candidates[region];

						g_nte_array_for_matching_unit_index = 0;
						g_matching_sequence_index = 0;
						g_matching_order_size_of_core = 0;

						// BFS method
						if (!DYNAMIC) {
							buildBFSCoreQueryTree();
							buildBFSTreeCPI(root_cand_id);
							//backwardPrune();
							generateInexactMatchingOrder();
						}
						// DYNAMIC method
						else {
							buildDynamicTreeCPI(root_cand_id);
							for (int ii = 0; ii < 20; ii++) {
								// backwardPrune();

								// if (is_contain_empty_indexset()) {
								//	cout << "empty" << endl;
								//	break;
								//}
							}

							// buildCoreQueryTree();
							generateMatchingOrderByDynamic();

						}

						// print correct partial match
						cout << "Exact Partial Match: ";
						for (int step = 0; step < g_core_size; step++) {
							CPINode* tmp_node = &indexSet[step];
							// cout << tmp_node << endl;
							// cout << tmp_node->size << endl;
							if (tmp_node->size == 1) {
								cout << step << ":" << tmp_node->candidates[0] << " ";
								// cout << step << ":" << tmp_node->candidates << " ";
							}
						}
						cout << endl;




						/*if (g_core_size != g_cnt_node_query_graph) {
							continue;
						}*/

						// Dynamic method
						//forwardBuildQueue(root_cand_id);

						//buildDynamicTreeCPI(root_cand_id);
						//forwardBuild(root_cand_id);
						//buildCoreQueryTree();
						// build core query tree

						//cout << "forward Done" << endl;

						//cout << "backward Done" << endl;
						//buildSearchIterator();
						//cout << "build search iterator Done" << endl;
						//generateMatchingOrder();
						//generateInexactMatchingOrder();
						//generateMatchingOrderByCoreQueryTree();
						//matchingOrderLayer();
						//test1();
						//cout << "generateMatchingOrder Done" << endl;				
						//findAllMatching();

						find_inexact_result();

						int aaa = 0;
					}
				}
				// output_result();

				// / Call the output_result function and assign its return value to resultInfo
				ResultInfo resultInfo = output_result();

				// Access the pairs in resultInfo
				// std::vector<std::pair<int, int>> pairs = resultInfo.pairs;
				std::vector<std::pair<int, int>> pairs = std::move(resultInfo.pairs);
				std::cout << "Size: " << pairs.size() << ", Capacity: " << pairs.capacity() << std::endl;


				// Now you can work with the pairs in your main function

				for (const auto& pair : pairs) {
					int result_1 = pair.first;
					int result_2 = pair.second;
					// Do something with i and result_i
					// cout << result_1 << endl;
					// cout << result_2 << endl;
				}

				// Define the folder path where you want to save the file
				std::string folderPath = "E:\\Master Thesis\\comparisons\\GFinder\\test_dataset\\" + dataset_n + "\\results_noise_fea_" + mean_fea + "_" + std_fea + "\\"; // Replace with your folder path
				// std::string folderPath = "E:\\Master Thesis\\comparisons\\GFinder\\gfinder\\GFinder-master\\GFinder-master\\test_dataset\\results_noise_fea_0_0\\";

				// namespace fs = std::filesystem;

				// Combine the folder path and the file name
				std::string filePath = folderPath + "results" + strNumber + "_" + strNumber2 + ".txt";


				// Saving pairs to the specified folder and file
				std::ofstream outputFile(filePath);
				if (outputFile.is_open()) {
					for (const auto& pair : pairs) {
						int i = pair.first;
						int result_i = pair.second;
						outputFile << i << "," << result_i << "\n";
					}
					outputFile.close();
					std::cout << "Pairs saved to '" << filePath << "'." << std::endl;
				}
				else {
					std::cerr << "Failed to open the output file." << std::endl;
				}

				cout << "ATEENTION: Cost: " << (clock() - g_clock) * 1.0 / CLOCKS_PER_SEC << endl;

				pairs.clear();        // Clear all elements
				pairs.shrink_to_fit(); // Request to reduce capacity to fit size

				std::cout << "Size: " << pairs.size() << ", Capacity: " << pairs.capacity() << std::endl;


				// At the end of the function:
				resetGlobalVariables_1();

				// At the end of the function:
				resetGlobalVariables_2();
				
			}

		}
		

	}
	//fin_query.close();
	system("pause");
    return 0;
}

