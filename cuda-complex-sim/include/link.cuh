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

#ifndef LINK_CUH_
#define LINK_CUH_

#include <stdint.h>

#include "parameters.cuh"



struct __align__(8) Link {
intptr_t target;
};

/* Add a new link between a source node and a target node.
 * WARNING: it doesn't allocate a supplementary array if the node has more than average_links_number links.
 */

__device__ inline bool addLink(int32_t source_id, int32_t target_id)
{
	for(uint32_t i=0; i<average_links_number; i++)
	{
		if(links_targets_array[source_id*average_links_number+i].target==-1)
		{
			links_targets_array[source_id*average_links_number+i].target=target_id;
			return true;
		}
	}
	return false;
}

/* Check if node is linking target
 * WARNING: this function acts on GLOBAL memory and doesn't perform the check on supplementary links array
 */

__device__ inline bool isLinked(int32_t node, uint32_t target)
{
	uint32_t i;
	for(i=node*average_links_number; i<(node+1)*average_links_number; i++)
	{
		if(links_targets_array[i].target==target)
		{
			return true;
		}
	}
	return false;
}




/* Check if node is linking target
 * WARNING: this function acts on SHARED memory and doesn't perform the check on supplementary links array
 */
__device__ inline bool isLinked(uint32_t target, Link* targets_tile)
{
	uint32_t i;
	for(i=threadIdx.x*average_links_number; i<(threadIdx.x+1)*average_links_number; i++)
	{
		if(targets_tile[i].target==target)
		{
			return true;
		}
	}
	return false;
}


__device__ inline uint8_t addLink(int32_t source_id, int32_t target_id, float weight)
{
	uint16_t i;

	#pragma unroll

	for(i=0;i<average_links_number;i++){
		if(links_targets_array[threadIdx.x*average_links_number+i].target==-1)		//there is no need to allocate supplementary space
			{
			links_targets_array[threadIdx.x*average_links_number+i].target=target_id;
			return 1;
		}
	}

	Link* temp;
	if(links_targets_array[threadIdx.x*average_links_number+i-2].target!=-2)		//supplementary space has not been allocated yet
	{
		temp = (Link*)malloc(supplementary_links_array_size*sizeof(Link));

				/* Initializes the supplementary array to -1 */

				uint16_t j=0;
				Link init;
				init.target=-1;
				while(j<supplementary_links_array_size)
				{
					temp[j]=init;
					j++;
				}

				// Copy neighbours_tile's last 2 elements in the first 2 elements of temp,
				// adds the new link and finally save temp's address in neighbours_tile

				temp[0]=links_targets_array[threadIdx.x*average_links_number+i-2];
				temp[1]=links_targets_array[threadIdx.x*average_links_number+i-1];
				temp[2].target=target_id;

				links_targets_array[threadIdx.x*average_links_number+i-1].target=(intptr_t)temp;   		// supplementary neighbors pointer is stored in last position
				links_targets_array[threadIdx.x*average_links_number+i-2].target=-2;					//-2 is the marker that tell us that this node has allocated space for its neighbors list
				return 2;
	}
	else  								//supplementary space has been allocated previously
	{
		temp=(Link*)links_targets_array[threadIdx.x*average_links_number+i-1].target;

		#pragma unroll
		for(i=0;i<supplementary_links_array_size;i++)
		{
			if(temp[i].target!=-1)
			{
				temp[i].target=target_id;
				return 3;
			}
		}
		return 4;		//an error has occurred
	}
}

__device__ inline void removeLink(uint16_t index)
{
	links_targets_array[index].target=-1;
}

#endif /* LINK_CUH_ */

