#ifndef LINK_HPP_
#define LINK_HPP_
#include "parameters.hpp"

struct Node;										//forward declaration

struct Link{
	Node* target;
	float weight;									//TODO distance?
	__host__ __device__ Link(Node* tar, float w)
	{
		target=tar;
		weight=w;
	}
};
#endif
