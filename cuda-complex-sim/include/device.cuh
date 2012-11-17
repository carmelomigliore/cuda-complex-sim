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
#include "node.hpp"
#include "link.hpp"
#include "parameters.hpp"
#include "message.hpp"
#include "task.hpp"
#include "barabasi_game.hpp"
#include "templates.hpp"

using namespace std;


/*Used to copy a piece of an array from Host to Device
* h_source is the address of the host source array at the index "start"
 */
template <typename T>
__host__ inline void copyToDevice(T* h_source, T* d_target, int32_t start, int32_t size ){
	if(cudaMemcpy(d_target,h_source,(size*sizeof(T)), cudaMemcpyHostToDevice)!= cudaSuccess){
		cerr << "\nCouldn't copy date to Device from Host";
		}
	}

/*Used to copy a piece of an array from Device to Host
* h_target is the address of the target host array at the index "start"
 */
template <typename T>
__host__ inline void copyFromDevice(T* d_source, T* h_target,int32_t start, int32_t size){
	if(cudaMemcpy(h_target,d_source,(size*sizeof(T)), cudaMemcpyDeviceToHost) != cudaSuccess){
		cerr << "\nCouldn't copy date to Host From Device";
	}
}

/*
 * Initializes all data structures on device. Preallocate all needed memory.
 */


__host__ bool allocateDataStructures(bool** nodes_dev, float2** nodes_coord_dev, task_t** task_dev, task_arguments** task_args_dev, Link** links_target_dev, message_t** inbox_dev, message_t** outbox_dev, int32_t** inbox_counter_dev, int16_t** outbox_counter_dev, curandState** d_state, uint32_t** barabasi_links, int32_t** actives_dev, uint32_t max_nodes, uint8_t avg_links, uint32_t active_size, uint16_t supplementary_size, uint16_t max_messages, uint16_t barabasi_initial_nodes){

	/* allocate nodes array */

	if(cudaMalloc((void**)nodes_dev,max_nodes*sizeof(bool))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device 1";
		return false;
	}
	printf("\nAllocated %d bytes",max_nodes*sizeof(bool));
	/*if(cudaMalloc((void**)nodes_coord_dev,max_nodes*sizeof(float2))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device 2";
			return false;
	}*/

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
	/*if(cudaMalloc((void**)links_weight_dev, max_nodes*avg_links*sizeof(float))!=cudaSuccess){
			cerr << "\nCouldn't allocate memory on device";
			return false;
	}*/

	/*if(cudaMalloc((void**)actives_dev, active_size*sizeof(int32_t))!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}*/


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
	uint32_t* count;
	if(cudaMalloc((void**)&count,sizeof(uint32_t))!=cudaSuccess)
	{
			cerr << "\nCouldn't allocate memory on device 11";
			return false;
	}



	/* copy constants to device memory */

	if(cudaMemcpyToSymbol(fail_count, &count, sizeof(uint32_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
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
	if(cudaMemcpyToSymbol(supplementary_links_array_size, &supplementary_size, sizeof(uint16_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(message_queue_size, &max_messages, sizeof(uint16_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(initial_nodes, &barabasi_initial_nodes, sizeof(uint16_t),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}


	/* copy arrays' addresses to device memory */

	if(cudaMemcpyToSymbol(nodes_array, nodes_dev, sizeof(bool*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	if(cudaMemcpyToSymbol(nodes_coord_array, nodes_coord_dev, sizeof(float2*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
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
	/*if(cudaMemcpyToSymbol(outbox_array, outbox_dev, sizeof(message_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(message_counter, inbox_counter_dev, sizeof(int32_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}
	if(cudaMemcpyToSymbol(outbox_counter, outbox_counter_dev, sizeof(int16_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}*/
	if(cudaMemcpyToSymbol(links_linearized_array, barabasi_links, sizeof(uint32_t*),0,cudaMemcpyHostToDevice)!=cudaSuccess){
		cerr << "\nCouldn't allocate memory on device";
		return false;
	}

	/* Success! */
	return true;
}






__global__ void test (){

	uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
	uint32_t ltid = threadIdx.x;
	if(tid==0)
	{
		printf("\nNodes: %x, Coord: %x", nodes_array, nodes_coord_array);
	}
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

	addNode(tid,coord);
	__syncthreads();
	//printf("Nodo n° %d creato\n", tid);

	extern __shared__ Link cache[];

	Link* targets_tile= copyToTileReadAhead<Link> (links_targets_array,cache, 0,5);
	__syncthreads();

	if(tid==0)
	{
		uint8_t i=0;

	}

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



__global__ void scale_free(curandState *state)
{
		uint32_t gtid = threadIdx.x + blockIdx.x*blockDim.x;
		Link init;
		init.target=-1;
		//init.weight=-1;
		//init.to_remove=false;
		initArray<bool>(false,nodes_array,max_nodes_number);
		initArray<Link>(init, links_targets_array, max_nodes_number*average_links_number);
		//initArray<task_t>(NULL, task_array, max_nodes_number);
		//initArray<int32_t>(0,message_counter,max_nodes_number);
		//initArray<int16_t>(0,outbox_counter,max_nodes_number);
		*fail_count=0;
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

__global__ void reset()
{
	*fail_count=0;
}

#endif /* DEVICE_CUH_ */





