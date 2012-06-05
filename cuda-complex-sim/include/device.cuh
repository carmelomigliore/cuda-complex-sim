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
 * License along with Cuda-complex-sim.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DEVICE_CUH_
#define DEVICE_CUH_

#include <iostream>
#include <stdint.h>

#include "node.hpp"
#include "node_resource.hpp"
#include "link.hpp"
#include "parameters.hpp"
#include "message.hpp"

using namespace std;


/*
 * Initializes all data structures on device. Preallocate all needed memory. TODO: write a template kernel that initialize all the arrays.
 */


__host__ bool allocateDataStructures(bool* nodes, float* nodes_x, float* nodes_y, int32_t* links_target, float* links_weight, int32_t* actives, uint32_t max_nodes, uint8_t max_links, uint32_t active_size){

	/* allocate nodes array */

	if(cudaMalloc((void**)&nodes,max_nodes*sizeof(bool))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)&nodes_x,max_nodes*sizeof(float))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}
	if(cudaMalloc((void**)&nodes_y,max_nodes*sizeof(float))!=cudaSuccess){
				cerr << "\nCouldn't allocate memory on device";
				return false;
	}


	/* allocate links arrays */

	if(cudaMalloc((void**)&links_target, max_nodes*max_links*sizeof(int32_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)&links_weight, max_nodes*max_links*sizeof(float))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}
	if(cudaMalloc((void**)&actives, active_size*sizeof(int32_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy constants to device memory */

	if(cudaMemcpyToSymbol(max_nodes_number, &max_nodes, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(max_links_number, &max_links, sizeof(uint8_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(active_nodes_array_size, &active_size, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy arrays' addresses to device memory */

	if(cudaMemcpyToSymbol(nodes_array, &nodes, sizeof(bool*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(nodes_coord_x_array, &nodes_x, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(nodes_coord_y_array, &nodes_y, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(links_targets_array, &links_target, sizeof(int32_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(links_weights_array, &links_weight, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	/* Success! */
	return true;
}

#endif /* DEVICE_CUH_ */




