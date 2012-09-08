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

#ifndef NODE_HPP_
#define NODE_HPP_

#include <stdint.h>


#include "math.h"
#include "parameters.hpp"
#include "node_resource.hpp"
#include "link.hpp"



	/*
	 * Calculate Euclidean distance of the node from the given coordinates (uses intrinsics functions)
	 */

__device__ inline float calculateDistance(float2 c1, float2 c2){
	return __fsqrt_rn(__powf(c2.x-c1.x,2)+__powf(c2.y-c1.y,2));
}

	/*
	 * 	Create a node and add it to the nodes array. Node creation can be done in parallel.
	 */

 __device__ inline void addNode(int32_t id, float2 coord){  //TODO la lista dei nodi deve essere costruita in modo ordinato (i vicini devono essere vicini anche come indici)
	 nodes_array[id]=true;
	 nodes_coord_array[id]=coord;

	 /*__syncthreads();

	 if(id!=0)
	 {
		 links_targets_array[id*average_links_number].target=id-1;
		 links_targets_array[id*average_links_number].weight=calculateDistance(coord,nodes_coord_array[id-1]);
		 links_targets_array[id*average_links_number].to_remove=false;
		 //printf("\nenzos : %d", links_targets_array[id*average_links_number].target);
	 }
	 else
	 {
		 links_targets_array[0].target=1;
		 links_targets_array[0].weight=calculateDistance(coord,nodes_coord_array[1]);
		 links_targets_array[0].to_remove=false;
	 }
	 */
 }

#endif /* NODES_HPP_ */
