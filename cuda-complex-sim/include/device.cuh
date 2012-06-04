#ifndef DEVICE_CUH_
#define DEVICE_CUH_

#include <iostream>
#include <stdint.h>

#include "node.hpp"
#include "node_resource.hpp"
#include "link.hpp"
#include "parameters.hpp"
#include "message.hpp"

using namespace std;


/*
 * Initializes all data structures on device. Preallocate all needed memory. TODO: write a template kernel that initialize all the arrays.
 */

__host__ bool allocateDataStructures(Node** nodes_dev_array, Link** links_dev_array, Node*** active_node_dev, Message** message_dev_array, int max_nodes, short max_links, int active_size, short message_buffer){

	if(cudaMemcpyToSymbol(max_nodes_number, &max_nodes, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(max_links_number, &max_links, sizeof(uint8_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(active_nodes_array_size, &active_size, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(max_links_number, &max_links, sizeof(uint8_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)nodes_dev_array,max_nodes*sizeof(Node))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)links_dev_array, max_nodes*max_links*sizeof(Link))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void***)active_node_dev, active_size*sizeof(Link*))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)message_dev_array, max_nodes*message_buffer*sizeof(Message))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	return true;
}



#endif /* DEVICE_CUH_ */




