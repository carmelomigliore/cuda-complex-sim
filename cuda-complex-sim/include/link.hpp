#ifndef LINK_HPP_
#define LINK_HPP_

#include <stddef.h>

struct Node;										//forward declaration

struct Link{
	Node* target;
	float weight;
	__host__ __device__ Link(){
		target=NULL;
		weight=0;
	}
	__host__ __device__ Link(Node* tar, float w)
	{
		target=tar;
		weight=w;								//link's weight is the Euclidean distance between a node and its neighbour
	}
};
#endif
