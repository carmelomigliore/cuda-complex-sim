/* Copyright (C) 2012 Carmelo Migliore, Fabrizio Gueli
 *
 * This file is part of Cuda-complex-sim
 *
 * Cuda-complex-sim is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * Cuda-complex-sim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LINK_HPP_
#define LINK_HPP_

#include <stdint.h>

#include "parameters.hpp"



typedef struct __align__(16) enzo {
intptr_t target;
float weight;
bool to_remove;
}Link;

/*
 * Add a new link between a source node and a target node.
 * To be used ONLY after neighbors array has been copied in a tile in shared memory.
 */

//TODO Convertire link in una struct in modo da poter sfruttare meglio la shared memory

__device__ inline uint8_t addLink(int32_t source_id, int32_t target_id, float weight, Link* neighbors_tile)
{
	uint16_t i;

	#pragma unroll

	for(i=0;i<average_links_number;i++){
		if(neighbors_tile[threadIdx.x*average_links_number+i].target==-1)		//there is no need to allocate supplementary space
			{
			neighbors_tile[threadIdx.x*average_links_number+i].target=target_id;
			neighbors_tile[threadIdx.x*average_links_number+i].weight=weight;
			return 1;
		}
	}


	Link* temp;
	if(neighbors_tile[threadIdx.x*average_links_number+i-2].target!=-2)		//supplementary space has not been allocated yet
	{
		temp = (Link*)malloc(supplementary_links_array_size*sizeof(Link));
		temp[0].target=neighbors_tile[threadIdx.x*average_links_number+i-2].target;
		temp[0].weight=neighbors_tile[threadIdx.x*average_links_number+i-2].weight;
		temp[1].target=neighbors_tile[threadIdx.x*average_links_number+i-1].target;
		temp[1].weight=neighbors_tile[threadIdx.x*average_links_number+i-1].weight;
		temp[2].target=target_id;
		temp[2].weight=weight;

		neighbors_tile[threadIdx.x*average_links_number+i-1].target=(intptr_t)temp;   		// supplementary neighbors pointer is stored in last position
		neighbors_tile[threadIdx.x*average_links_number+i-2].target=-2;					//-2 is the marker that tell us that this node has allocated space for its neighbors list
		return 2;
	}
	else  								//supplementary space has been allocated previously
	{
		temp=(Link*)neighbors_tile[threadIdx.x*average_links_number+i-1].target;

		#pragma unroll
		for(i=0;i<supplementary_links_array_size;i++)
		{
			if(temp[i].target!=-1)
			{
				temp[i].target=target_id;
				temp[i].weight=weight;
				return true;
			}
		}
		return 3;		//an error has occurred
	}
}

__device__ inline void removeLink(uint16_t index, Link* neighborsTile)
{
	neighborsTile[index].target=-1;
	neighborsTile[index].weight=-1;
}

#endif /* LINK_HPP_ */

