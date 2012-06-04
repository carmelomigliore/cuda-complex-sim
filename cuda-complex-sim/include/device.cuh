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

/*
 * Create the first two nodes of the graph and connect them to each other
 */

__device__ void initGraph(float x0, float y0, NodeResource nr0,float x1,float y1, NodeResource nr1 ){
	Node node0(0,x0,y0,nr0);
	Node node1(1,x1,y1,nr1);
	nodes_dev_array[0]=node0;
	nodes_dev_array[1]=node1;
	links_dev_array[0].target=&nodes_dev_array[1];
	links_dev_array[0].weight=node0.calculateDistance(node1.x,node1.y);
	links_dev_array[max_links_number].target=&nodes_dev_array[0];   						//In this case links_dev_array's index should be 1*max_links_numbers but we omitted it
	links_dev_array[max_links_number].weight=node1.calculateDistance(node0.x,node0.y);

}



#endif /* DEVICE_CUH_ */




