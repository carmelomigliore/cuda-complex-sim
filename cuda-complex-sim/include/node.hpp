#ifndef NODE_HPP_
#define NODE_HPP_

#include <stdint.h>


#include "math.h"
#include "link.hpp"
#include "parameters.hpp"
#include "node_resource.hpp"


/*
 * Class that implements a node's concept.
 */


struct Node {
	uint32_t id;     												// node's id
	float x, y;													// node's coordinates
	NodeResource res;											// node's resources

	/*
	 * Node constructor
	 */
	__device__ Node(uint32_t node_index, float coord_x, float coord_y, NodeResource r){
		id=node_index;
		x=coord_x;
		y=coord_y;
		res=r;
	}

	/*
	 * Calculate Euclidean distance of the node from the given coordinates
	 */
	__device__ float calculateDistance(float targ_x, float targ_y){
		return sqrtf(powf(x-targ_x,2)+powf(y-targ_y,2));
	}

	/*
	 * 	Create a node that is NOT the first of the graph and add it to the nodes array. Node creation can be done in parallel.
	 */

	__device__ static void addNode(unsigned int id, float x, float y, NodeResource nr, Node* node_array_dev, Link* link_array_dev){
		Node node(id,x,y,nr);
		node_array_dev[id]=node;
		Node* neighbour;
		if(id!=0){
			neighbour=&node_array_dev[id-1];
			link_array_dev[id*max_links_number].target=neighbour;											// first neighbour is assigned to connect the node to the net. It is the the previous node on the nodes array.
			link_array_dev[id*max_links_number].weight=node.calculateDistance(neighbour->x,neighbour->y);
		}
	}
	/*
	 * Establish a new link TODO: controllare se può essere inline (perché c'è il while)
	 */
	__device__ bool addLink(Node* trg, float distance);
	/*
	__device__ message_t readMessage();				// da implementare qui per essere inline
	__device__ bool sendMessage(message_t m, Vertex* target);
	__device__ void discovery();
	*/
};
#endif
