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


__host__ bool allocateDataStructures(bool** nodes_dev, float2** nodes_coord_dev, Link** links_target_dev, int32_t** actives_dev, uint32_t max_nodes, uint8_t avg_links, uint32_t active_size){

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

	if(cudaMalloc((void**)links_target_dev, max_nodes*avg_links*sizeof(Link))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	/*if(cudaMalloc((void**)links_weight_dev, max_nodes*avg_links*sizeof(float))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}*/

	/*if(cudaMalloc((void**)actives_dev, active_size*sizeof(int32_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}*/


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

	if(cudaMemcpyToSymbol(nodes_array, nodes_dev, sizeof(bool*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	printf("Nodes_mall: %x, Links_mall: %x", *nodes_dev, *links_target_dev);

	if(cudaMemcpyToSymbol(nodes_coord_array, nodes_coord_dev, sizeof(float2*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(links_targets_array, links_target_dev, sizeof(Link*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	/*if(cudaMemcpyToSymbol(links_weights_array, links_weight_dev, sizeof(float*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}*/

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
 * Used to copy a piece of an array from global memory INTO a tile in shared memory
 * Nota bene: adesso funziona, per favore di cristo non toccarla più.
 */

template <typename T>
__device__ inline void copyToTile(T* source, T* tile, uint16_t start, uint16_t elements_per_thread){		//elements_per_thread indica quanti elementi deve copiare ciascun thread. Così ad esempio se è uguale a 5 e ogni blocco è formato da 10 thread, in totale verranno copiati nella shared memory 50 elementi
	uint16_t tid=threadIdx.x; 																				//thread index in this block
	while(tid<blockDim.x*elements_per_thread)
	{
		tile[tid]=source[start+tid+blockIdx.x*blockDim.x*elements_per_thread];
		tid+=blockDim.x;
	}
}
/*
 * Used to copy back from a tile in shared memory to an array in global memory
 */

template <typename T>
__device__ inline void copyFromTile(T* source, T* tile, uint16_t start, uint16_t elements_per_thread){
	uint16_t tid=threadIdx.x; 																				//thread index in this block
	while(tid<blockDim.x*elements_per_thread)
	{
		source[start+tid+blockIdx.x*blockDim.x*elements_per_thread]=tile[tid];
		tid+=blockDim.x;
	}
};


__global__ void test (){

	uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid==0)
	{
		printf("\nNodes: %x, Coord: %x", nodes_array, nodes_coord_array);
	}
	float2 coord;
	coord.x=tid*3;
	coord.y=tid*7;

	Link init;
	init.target=-1;
	init.weight=-1;
	init.to_remove=false;
	initArray<bool>(false,nodes_array,10000);
	initArray<Link>(init, links_targets_array, 50000);
	__syncthreads();

	addNode(tid,coord);
	__syncthreads();
	//printf("Nodo n° %d creato\n", tid);

	extern __shared__ Link targets_tile[];

	copyToTile<Link> (links_targets_array,targets_tile, 0,5);
	__syncthreads();

	if(tid==32)
	{
		uint8_t i=0;
		while(i<25)
		{
			printf("\nCristoCristo %d",i);
			printf("\nCristenzo %d", targets_tile[i].target);
			printf("\nDiocristo %d", links_targets_array[i+tid*5].target);
			i++;
		}
	}

	if(tid==32)
		{
			printf("\nCristogesu %d", addLink(tid,2, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,3, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,4, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,5, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,6, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,7, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,8, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,9, 100, targets_tile));
			printf("\nCristogesu %d", addLink(tid,10, 100, targets_tile));
		}

	/*uint8_t i = 0;
	while(i<average_links_number)
	{
		printf("\nLink del nodo %d: %d",tid, targets_tile[tid*average_links_number+i]);
		i++;
	}*/
}

#endif /* DEVICE_CUH_ */




