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

#include "curand_kernel.h"
#include "node.cuh"
#include "link.cuh"
#include "parameters.cuh"
#include "message.cuh"
#include "task.cuh"
#include "barabasi_game.cuh"
#include "templates.cuh"
#include "h_barabasi_game.hpp"
#include "h_parameters.hpp"







/*
 * Function used to copy the Supplementary Arrays from host to Device
 * WARNING: TO BE USED ONLY AFTER copying link's array to Device
 */
__host__ void copySupplementaryArrayToDevice(Link* d_links_array)
{
	Link* temp;
	Link* dev;
	uint32_t c= 0;
	for(uint32_t i=0; i<h_max_nodes_number;i++)
		{
			for(uint16_t j=0; j<h_average_links_number;j++)
			{
				if(h_links_target_array[i*h_average_links_number+j].target==-2)
				{
					temp= (Link*)h_links_target_array[i*h_average_links_number+j+1].target;
					h_addr.push_back((intptr_t)temp);
					if(d_addr.empty())
					{
						if(cudaMalloc((void**)&dev,h_supplementary_links_array_size*sizeof(Link))!=cudaSuccess)
							cerr << "\nCouldn't allocate memory on device";
						if(cudaMemcpy(dev,temp,h_supplementary_links_array_size*sizeof(Link), cudaMemcpyHostToDevice) != cudaSuccess)
						{
							cerr << "\nCouldn't copy date to Host From Device";
						}
						if(cudaMemcpy(&d_links_array[i*h_average_links_number+j+1].target,&dev,h_supplementary_links_array_size*sizeof(Link), cudaMemcpyHostToDevice) != cudaSuccess)
						{
							cerr << "\nCouldn't copy date to Host From Device";
						}

					}
					else
					{
						if(cudaMemcpy((Link*)d_addr[c],temp,h_supplementary_links_array_size*sizeof(Link), cudaMemcpyHostToDevice) != cudaSuccess)
						{
							cerr << "\nCouldn't copy date to Host From Device";
						}
						if(cudaMemcpy(&d_links_array[i*h_average_links_number+j+1].target,&d_addr[c],h_supplementary_links_array_size*sizeof(Link), cudaMemcpyHostToDevice) != cudaSuccess)
						{
							cerr << "\nCouldn't copy date to Host From Device";
						}
						d_addr.erase(d_addr.begin()+c);
						c++;


					}

				}

			}
		}
}

/*
 * Function used to copy the Supplementary Arrays from Device to Host
 * WARNING: TO BE USED ONLY AFTER copying link's array from Device
 */

__host__ void copySupplementaryArrayFromDevice()
{
	Link* temp;
	Link* host;
	uint32_t c = 0;
	for(uint32_t i=0; i<h_max_nodes_number;i++)
		{
			for(uint16_t j=0; j<h_average_links_number;j++)
			{
				if(h_links_target_array[i*h_average_links_number+j].target==-2)
				{
					temp= (Link*)h_links_target_array[i*h_average_links_number+j+1].target;
					d_addr.push_back((intptr_t)temp);
					if(!h_addr.empty())
					{
						if(cudaMemcpy((Link*)h_addr[c],temp,h_supplementary_links_array_size*sizeof(Link), cudaMemcpyDeviceToHost) != cudaSuccess)
						{
							cerr << "\nCouldn't copy date to Host From Device";
						}
						h_links_target_array[i*h_average_links_number+j+1].target= (intptr_t)h_addr[c];
						h_addr.erase(h_addr.begin()+c);
						c++;

					}
					else
					{
						host = (Link*)malloc(h_supplementary_links_array_size*sizeof(Link));
						h_links_target_array[i*h_average_links_number+j+1].target= (intptr_t)host;
					}


				}
			}

		}
}


/*
 * Initializes all data structures on device. Preallocate all needed memory.
 */


__host__ bool allocateDataStructures(n_attribute** pr_attr,bool** nodes_dev, task_t** task_dev, task_arguments** task_args_dev, Link** links_target_dev, message_t** inbox_dev, curandState** d_state, uint32_t** barabasi_links, uint32_t max_nodes, uint8_t avg_links, uint16_t supplementary_size, uint16_t barabasi_initial_nodes){

	/* allocate nodes array */

	if(cudaMalloc((void**)pr_attr,max_nodes*sizeof(n_attribute))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device 0";
			return false;
		}

	if(cudaMalloc((void**)nodes_dev,max_nodes*sizeof(bool))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device 1";
		return false;
	}
	printf("\nAllocated %d bytes",max_nodes*sizeof(bool));

	/*Allocate task arrays */

	/*if(cudaMalloc((void**)task_dev,sizeof(task_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device 3";
		return false;
	}

	if(cudaMalloc((void**)task_args_dev,max_nodes*sizeof(task_arguments))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device 4";
			return false;
		}*/


	/* allocate links arrays */

	if(cudaMalloc((void**)links_target_dev, max_nodes*avg_links*sizeof(Link))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device 5";
		return false;
	}
	printf("\nAllocated %d bytes",max_nodes*avg_links*sizeof(Link));


	/* Allocate messages arrays */

	if(cudaMalloc((void**)inbox_dev, max_nodes*sizeof(message_t))!=cudaSuccess)
	{
		cerr << "\nCouldn't allocate memory on device 6";
		return false;
	}


	/* Allocate curand seeds array */

	if(cudaMalloc((void**)d_state, BLOCKS*THREADS_PER_BLOCK*sizeof(curandState))!=cudaSuccess)
	{
		cerr << "\nCouldn't allocate memory on device 10";
		return false;
	}

	/*Barabasi parameters */

	if(cudaMalloc((void**)barabasi_links,(barabasi_initial_nodes*(barabasi_initial_nodes-1)*2+(max_nodes-barabasi_initial_nodes)*avg_links*2)*sizeof(uint32_t))!=cudaSuccess)
	{
		cerr << "\nCouldn't allocate memory on device 11";
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

	if(cudaMemcpyToSymbol(supplementary_links_array_size, &supplementary_size, sizeof(uint16_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(initial_nodes, &barabasi_initial_nodes, sizeof(uint16_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy arrays' addresses to device memory */

	if(cudaMemcpyToSymbol(nodes_programattr_array, pr_attr, sizeof(n_attribute*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(nodes_array, nodes_dev, sizeof(bool*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	if(cudaMemcpyToSymbol(task_array, task_dev, sizeof(task_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(task_arguments_array, task_args_dev, sizeof(task_arguments*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(links_targets_array, links_target_dev, sizeof(Link*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(cstate, d_state, sizeof(curandState*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(message_array, inbox_dev, sizeof(message_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(links_linearized_array, barabasi_links, sizeof(uint32_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	/* Success! */
	return true;
}



__global__ void test (){

	uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
	if(tid==0)
	{
	//	printf("\nNodes: %x, Coord: %x", nodes_array, nodes_coord_array);
	}
	float2 coord;
	coord.x=tid*3;
	coord.y=tid*7;

	Link init;
	init.target=-1;
	initArray<bool>(false,nodes_array,30000);
	initArray<Link>(init, links_targets_array, 30000*5);
	initArray<task_t>(NULL, task_array, 30000);
	__syncthreads();

	addNode(tid,coord);
	__syncthreads();
	//printf("Nodo n° %d creato\n", tid);

	extern __shared__ Link cache[];

	Link* targets_tile= copyToTileReadAhead<Link> (links_targets_array,cache, 0,5);
	__syncthreads();


	/*uint8_t i = 0;
	while(i<average_links_number)
	{
		printf("\nLink del nodo %d: %d",tid, targets_tile[tid*average_links_number+i]);
		i++;
	}*/
}

__device__ bool simpleTask(void * in, void** out)
{
	float* enzo = (float*)in;
	printf("\nHello Qualcomm APQ8064 %f", threadIdx.x+ blockIdx.x*blockDim.x+*enzo);
	return true;
}

__global__ void taskTest()
{
	uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
	uint16_t tid = threadIdx.x;
	float2 coord;
	coord.x=tid*3;
	coord.y=tid*7;

	Link init;
	init.target=-1;
	//init.weight=-1;
	//init.to_remove=false;
	initArray<bool>(false,nodes_array,30000);
	initArray<Link>(init, links_targets_array, 30000*5);
	initArray<task_t>(NULL, task_array, 30000);
	__syncthreads();

	addNode(gtid,coord);
	__syncthreads();

	__shared__ task_arguments arg_cache [THREADS_PER_BLOCK];

	float* optimus = (float*)malloc(sizeof(float));
	*optimus=0.123;
//	task_arguments init2; init2.in=(void*)optimus; init2.out=NULL;

	while(nodes_array[gtid]==true)
	{
	//	assignTask(gtid, simpleTask, init2);
		__syncthreads();	//da controllare
		copyToTile <task_arguments> (task_arguments_array, arg_cache, 0, 1, 0);
		task_array[gtid](arg_cache[tid].in, &arg_cache[tid].out);
		gtid+= blockDim.x * gridDim.x;
	}
}

__global__ void init_stuff(curandState *state, unsigned long long seed) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
curand_init(seed, idx, 0, &state[idx]);
}

__global__ void init_data()
{
			Link init;
			init.target=-1;
			initArray<bool>(false,nodes_array,max_nodes_number);
			initArray<Link>(init, links_targets_array, max_nodes_number*average_links_number);
			//initArray<task_t>(NULL, task_array, max_nodes_number);

			__syncthreads();

}

__global__ void scale_free(curandState *state)
{
		uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
		Link init;
		init.target=-1;
		initArray<bool>(false,nodes_array,max_nodes_number);
		initArray<Link>(init, links_targets_array, max_nodes_number*average_links_number);
		//initArray<task_t>(NULL, task_array, max_nodes_number);
		__syncthreads();

		if(gtid==0)
		{
			barabasi_game(initial_nodes,average_links_number,max_nodes_number,state);
		}
}

__global__ void message_test()
{
	uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ Link tile [];
	while(gtid<max_nodes_number)
	{
		generateMessages(tile,max_nodes_number,gtid,30);
		gtid+=blockDim.x*gridDim.x;
	}
}
__global__ void message_test2nd()
{
	uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
	extern __shared__ Link targets_tile [];
	while(gtid<max_nodes_number)
	{
		checkInbox(targets_tile,gtid);
		gtid+=blockDim.x*gridDim.x;
	}
}

__global__ void print()
{
	printf("\n%d %d",max_nodes_number,average_links_number);
}


#endif /* DEVICE_CUH_ */





