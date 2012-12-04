/* Copyright (C) 2012 Fabrizio Gueli
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

#ifndef HOST_HPP_
#define HOST_HPP_

#include <iostream>
#include <stdint.h>
#include <stdio.h>

#include "h_node.hpp"
#include "h_link.hpp"
#include "h_parameters.hpp"
#include "h_templates.hpp"
#include "graph_transf.hpp"
#include "device.cuh"

using namespace std;

/*
 * Initializes all data structures on host. Preallocate all needed memory.
 */

__host__ bool h_allocateDataStructures(uint16_t supplementary_size, uint32_t max_nodes, uint8_t avg_links){

	h_max_nodes_number = max_nodes;
	h_average_links_number = avg_links;
	h_supplementary_links_array_size =supplementary_size;


	/* allocate nodes array */

	h_nodes_array = (bool*)malloc(h_max_nodes_number*sizeof(bool));
	if(h_nodes_array == NULL){
		cerr << "\nCouldn't allocate memory on host 1";
			return false;
		}
	printf("\nAllocated %d bytes",h_max_nodes_number*sizeof(bool));


	/* allocate links arrays */

	h_links_target_array = (Link*)malloc(h_max_nodes_number*h_average_links_number*sizeof(Link));
	if(h_links_target_array==NULL){
		cerr << "\nCouldn't allocate memory on host 3";
				return false;
	}
	printf("\nAllocated %d bytes",h_max_nodes_number*h_average_links_number*sizeof(Link));

	/* Allocate attributes arrays */


	h_nodes_programattr_array = (n_attribute*)malloc(h_max_nodes_number*sizeof(n_attribute));
	if(h_nodes_programattr_array == NULL){

			cerr << "\nCouldn't allocate memory on host 5";
			return false;
		}


	/* Success! */
	return true;
}

__host__ void startSimulation(Link* links,bool* nodes,uint16_t supplementary_size, uint32_t max_nodes, uint8_t avg_links,Graph g)
{
	Link init;
	init.target=-1;
	init_data<<<BLOCKS,THREADS_PER_BLOCK>>>();
	h_initArray<bool>(false,h_nodes_array,h_max_nodes_number);
	h_initArray<Link>(init, h_links_target_array, h_max_nodes_number*h_average_links_number);
	adjlistToCompactList(g);
	copyToDevice(nodes,h_nodes_array , 0, h_max_nodes_number );
	copyToDevice(links,h_links_target_array ,0, h_max_nodes_number*h_average_links_number );
}



#endif /* HOST_HPP_ */
