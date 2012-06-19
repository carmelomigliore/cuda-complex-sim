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

/*
 * Add a new link between a source node and a target node.
 * To be used ONLY after neighbors array has been copied in a tile in shared memory.
 */

__device__ inline bool addLink(int32_t source_id, int32_t target_id, float weight, intptr_t* neighbors_tile, float* weights_tile)
{
	uint16_t i;

	#pragma unroll

	for(i=0;i<average_links_number;i++){
		if(neighbors_tile[threadIdx.x*average_links_number+i]==-1){
			neighbors_tile[threadIdx.x*average_links_number+i]=target_id;
			weights_tile[threadIdx.x*average_links_number+i]=weight;
			return true;
		}
	}
	intptr_t* temp;
	float* tmpweight;
	if(neighbors_tile[threadIdx.x*average_links_number+i-3]!=-2)		//supplementary space has not been allocated yet
	{
		temp = (intptr_t*)malloc(200*sizeof(intptr_t));
		tmpweight = (float*)malloc(200*sizeof(float*));
		temp[0]=neighbors_tile[threadIdx.x*average_links_number+i-3];
		temp[1]=neighbors_tile[threadIdx.x*average_links_number+i-2];
		temp[2]=neighbors_tile[threadIdx.x*average_links_number+i-1];
		temp[3]=target_id;
		tmpweight[0]=weights_tile[threadIdx.x*average_links_number+i-3];
		tmpweight[1]=weights_tile[threadIdx.x*average_links_number+i-2];
		tmpweight[2]=weights_tile[threadIdx.x*average_links_number+i-1];
		tmpweight[3]=weight;

		neighbors_tile[threadIdx.x*average_links_number+i-1]=(intptr_t)temp;   		// supplementary neighbors pointer is stored in last position
		neighbors_tile[threadIdx.x*average_links_number+i-2]=(intptr_t)tmpweight;		// supplementary weights pointer is stored in second last position
		neighbors_tile[threadIdx.x*average_links_number+i-3]= -2;		 	//-2 is the marker that tell us that this node has allocated space for its neighbors list
		return true;
	}
	else  //supplementary space has been allocated previously
	{
		temp=(intptr_t*)neighbors_tile[threadIdx.x*average_links_number+i-1];
		tmpweight=(float*)neighbors_tile[threadIdx.x*average_links_number+i-2];

		#pragma unroll
		for(i=0;i<200;i++)
		{
			if(temp[i]!=-1)
			{
				temp[i]=target_id;
				tmpweight[i]=weight;
				return true;
			}
		}
		return false;		//an error has occurred
	}
}

__device__ inline void removeLink(uint16_t index, intptr_t* neighborsTile, float* weightsTile)
{
	neighborsTile[index]=-1;
	weightsTile[index]=-1;
}

#endif /* LINK_HPP_ */

