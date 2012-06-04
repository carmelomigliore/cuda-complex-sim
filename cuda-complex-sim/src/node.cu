#include <stdint.h>
#include <stddef.h>

#include "parameters.hpp"
#include "node.hpp"
#include "link.hpp"

__device__ bool Node::addLink(Node* trg, float distance){
		uint8_t i = 0;
		while(i<=max_links_number)
		{
			if(links_dev_array[id*max_links_number+i].target!=NULL){
				i++;
			}
			else {
				links_dev_array[id*max_links_number+i].target=trg;
				links_dev_array[id*max_links_number+i].weight=distance;
				return true;
			}
		}
		return false;
	}


