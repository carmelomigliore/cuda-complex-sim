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
 * To be used ONLY after neighboors array has been copied in a tile in shared memory.
 */

__device__ inline bool addLink(int32_t source_id, int32_t target_id, float weight, intptr_t* neighboors_tile, float* weights_tile)
{
	uint8_t i;

	#pragma unroll

	for(i=0;i<max_links_number;i++){

		if(neighboors_tile[threadIdx.x*max_links_number+i]==-1){
			neighboors_tile[threadIdx.x*max_links_number+i]=target_id;
			weights_tile[threadIdx.x*max_links_number+i]=weight;
			return true;
		}
	}
	return false;
}






#endif /* LINK_HPP_ */

