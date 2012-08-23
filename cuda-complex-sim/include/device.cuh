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
#include <stdio.h>

#include "node.hpp"
#include "link.hpp"
#include "parameters.hpp"
#include "message.hpp"

using namespace std;


/*
 * Initializes all data structures on device. Preallocate all needed memory. TODO: write a template kernel that initialize all the arrays.
 */


__host__ bool allocateDataStructures(bool** nodes_dev, float2** nodes_coord_dev, int32_t** links_target_dev, float** links_weight_dev, int32_t** actives_dev, uint32_t max_nodes, uint8_t avg_links, uint32_t active_size){

	/* allocate nodes array */

	if(cudaMalloc((void**)nodes_dev,max_nodes*sizeof(bool))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)nodes_coord_dev,max_nodes*sizeof(float2))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}


	/* allocate links arrays */

	if(cudaMalloc((void**)links_target_dev, max_nodes*avg_links*sizeof(intptr_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMalloc((void**)links_weight_dev, max_nodes*avg_links*sizeof(float))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}
	if(cudaMalloc((void**)actives_dev, active_size*sizeof(int32_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy constants to device memory */

	if(cudaMemcpyToSymbol(max_nodes_number, &max_nodes, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(average_links_number, &avg_links, sizeof(uint8_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(active_nodes_array_size, &active_size, sizeof(uint32_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy arrays' addresses to device memory */

	if(cudaMemcpyToSymbol(nodes_array, &nodes_dev, sizeof(bool*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(nodes_coord_array, &nodes_coord_dev, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(links_targets_array, &links_target_dev, sizeof(intptr_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(links_weights_array, &links_weight_dev, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	/* Success! */
	return true;
}



template <typename T>
__device__ inline void initArray(T initValue, T* devArray, uint32_t arrayDimension){
	uint32_t tid=threadIdx.x + blockIdx.x*blockDim.x;
	#pragma unroll
	while(tid<arrayDimension){
		devArray[tid]=initValue;
		tid+=gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z; //increments by the number of total threads
	}
};


/*
 * Used to copy a piece of an array into a tile (can be used to copy from or to SHARED memory)
 */

template <typename T>
__device__ inline void copyToTile(T* source, T* tile, uint16_t offset){
	uint32_t tid=threadIdx.x;
	#pragma unroll
	while(tid<offset){
		tile[tid]=source[tid];					//TODO verificare se è più veloce il while oppure memcpy
		tid+=blockDim.x*blockDim.y*blockDim.z;	//increments by the number of threads per block
	}
};


__global__ void test (){
	uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
	float2 coord;

	coord.x=tid*3;
	coord.y=tid*7;
	initArray<bool>(false,nodes_array,100000);
	addNode(tid,coord);
	printf("Nodo n° %d creato\n", tid);
}

#endif /* DEVICE_CUH_ */




