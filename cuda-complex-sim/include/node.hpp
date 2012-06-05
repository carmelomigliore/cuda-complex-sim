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



	/*
	 * Calculate Euclidean distance of the node from the given coordinates
	 */

	__device__ inline float calculateDistance(float x1, float y1, float x2, float y2){
		return sqrtf(powf(x2-x1,2)+powf(y2-y1,2));
	}

	/*
	 * Create the FIRST TWO nodes of the graph and connect them to each other
	 */

	__device__ inline void  initGraph(float x0, float y0, float x1,float y1){
		nodes_array[0]=true;
		nodes_array[1]=true;
		nodes_coord_x_array[0]=x0;
		nodes_coord_y_array[0]=y0;
		nodes_coord_x_array[1]=x1;
		nodes_coord_y_array[1]=y1;
		links_targets_array[0]=1;
		links_targets_array[1*max_links_number]=0;
		links_weights_array[0]=links_weights_array[1*max_links_number]= calculateDistance(x0,y0,x1,y1);
	}

	/*
	 * 	Create a node that is NOT the first of the graph (id MUST be >0) and add it to the nodes array. Node creation can be done in parallel.
	 */

 __device__ inline void addNode(int32_t id, float x, float y){
	 nodes_array[id]=true;
	 nodes_coord_x_array[id]=x;
	 nodes_coord_y_array[id]=y;
	 links_targets_array[id*max_links_number]=id-1;
	 links_weights_array[id*max_links_number]=calculateDistance(x,y,nodes_coord_x_array[id-1],nodes_coord_y_array[id-1]);
	}

#endif /* NODES_HPP_ */
