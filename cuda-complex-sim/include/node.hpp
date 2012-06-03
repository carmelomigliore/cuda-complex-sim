#ifndef NODE_HPP_
#define NODE_HPP_

#include "parameters.hpp"

/*
 * Class that implements a node's concept.
 */


struct Node {
	int id;     												// node's id
	int x, y;													// node's coordinates
	//message_t transit;										// message to be forwarded
	//Vertex* announced;
	int link_index;
	__host__ __device__  Node(int node_index, int coord_x, int coord_y)
	{
		id=node_index;
		x=coord_x;
		y=coord_y;
		link_index=max_links_number*id;
	}
	/*
	__device__ message_t readMessage();				// da implementare qui per essere inline
	__device__ bool sendMessage(message_t m, Vertex* target);
	__device__ void discovery();
	*/
};
#endif
