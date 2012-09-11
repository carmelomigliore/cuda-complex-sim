/* Copyright (C) 2012 Carmelo Migliore
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

#ifndef BARABASI_GAME_HPP_
#define BARABASI_GAME_HPP_

#include <stdlib.h>
#include "parameters.hpp"
#include "curand_kernel.h"

/* Generates a scale-free network using Barabasi's algorithm */

__device__ void barabasi_game(uint16_t initial_nodes, uint16_t links_number, uint32_t max_nodes, curandState *state)
{
	/* First we allocate links' linearized array, it will contain the target of every link
	 * It will be used to simulate probability.
	 * (initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)
	 */
	uint32_t counter=0; //total link counter
	uint32_t* links_linearized_array = (uint32_t*)malloc((initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)*sizeof(uint32_t));


	/* We create the first N nodes (==initial_nodes) and link all of them with one another*/

	float2 coord; coord.x=0; coord.y=0;  //fake coordinates
	uint32_t i=0;
	while(i<initial_nodes)
	{
		addNode(i, coord);
		i++;
	}

	uint32_t j;
	for(i=0; i<initial_nodes; i++)
	{
		for(j=0; j<initial_nodes; j++)
		{
			if(j==i)
			{
				continue; // Self-link are not allowed
			}
			else
			{
				addLink(i,j);
				links_linearized_array[counter]= i; //source and target are added to links_linearized_array
				links_linearized_array[counter+1]= j;
				counter+=2;
			}
		}
	}

	/* Now we add one node per time, and add all the links for that node, using Barabasi's algorithm */

	uint32_t random;
	uint32_t random_node;
	bool flag;
	for(i=initial_nodes; i< max_nodes; i++)
	{
		addNode(i, coord);
		for(j=0; j<links_number; j++)
		{
			flag=true;

			/* Let's see to what node the variable random corresponds,
			 * and if it is not already linked, link it. Otherwise, generates a new number.
			 */
			while(flag)
			{
				random = (uint32_t)(curand_uniform(state)*counter)%counter;		//generates a number between 0 and counter
				//printf("\nDino %d - %d", random, counter);
				random_node=links_linearized_array[random];
				if (!isLinked(i,random_node) && random_node!=i)
				{
					flag=false; //exit while
				}
			}
			addLink(i, random_node);
			links_linearized_array[counter]= i;			//Add the new link source and target to links_linearized_array
			links_linearized_array[counter+1]= random_node;
			counter+=2;
		}
	}
	__threadfence();

	for(int j=0; j<(initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2); j+=2)
	{
		printf("\nLink: %d - %d", links_linearized_array[j], links_linearized_array[j+1]);
	}
}


__device__ void  generateMessages(Link* targets_tile, int32_t nodes_number, int32_t this_node, uint16_t ttl)
{
	copyToTile<Link>(links_targets_array, targets_tile,(this_node/(gridDim.x*blockDim.x))*(gridDim.x*blockDim.x), average_links_number, 0); // la divisione restituisce la parte intera del quoziente
	uint32_t random_neighbour_idx;
	uint32_t random_receiver;
	message_t mex;
	random_receiver = (uint32_t)curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])%nodes_number;
	mex.sender=this_node; mex.receiver=random_receiver; mex.ttl=ttl;
	if(isLinked(this_node,random_receiver,targets_tile))
	{
		sendMessage(random_receiver,mex);
	}
	else
	{
		random_neighbour_idx = (uint32_t)curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])%average_links_number;
		sendMessage(targets_tile[threadIdx.x*average_links_number+random_neighbour_idx].target,mex);
	}
	__threadfence();
	__syncthreads();
}

__device__ void checkInbox(message_t* inbox_tile, message_t* outbox_tile, int32_t this_node, int32_t nodes_number)
{
	copyToTile<message_t>(message_array, inbox_tile,(this_node/(gridDim.x*blockDim.x))*(gridDim.x*blockDim.x), message_queue_size, 0);
	message_t mex;
	uint16_t out_counter=0;

	for(int i=0; i<message_counter[this_node];i++)
	{
		mex=inbox_tile[threadIdx.x*message_queue_size+i];
		if(mex.receiver==this_node)
		{
			printf("\nReceived!");
		}
		else if(mex.ttl==0)
		{
			printf("\nFine della corsa");
		}
		else
		{
			mex.ttl--;
			outbox_tile[threadIdx.x*message_queue_size+out_counter]=mex;
			out_counter++;
		}
	}
	message_counter[this_node]=0;
	outbox_counter[this_node]=out_counter;
	copyFromTile<message_t>(outbox_array,outbox_tile,(this_node/(gridDim.x*blockDim.x))*(gridDim.x*blockDim.x), message_queue_size,0);
	__threadfence();
	__syncthreads();
}

__device__ void sendOutbox(message_t* outbox_tile, Link* targets_tile, int32_t this_node)
{
	copyToTile<Link>(links_targets_array, targets_tile,(this_node/(gridDim.x*blockDim.x))*(gridDim.x*blockDim.x), average_links_number, 0); // la divisione restituisce la parte intera del quoziente
	message_t mex;
	uint32_t random_neighbour_idx;
	for(int i=0; i<outbox_counter[this_node];i++)
	{
		mex=outbox_tile[threadIdx.x*message_queue_size+i];
		if(isLinked(this_node,mex.receiver,targets_tile))
		{
			sendMessage(mex.receiver,mex);
		}
		else
		{
			random_neighbour_idx = (uint32_t)curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])%average_links_number;
			sendMessage(targets_tile[threadIdx.x*average_links_number+random_neighbour_idx].target,mex);
		}
	}
	outbox_counter[this_node]=0;
	__threadfence();
	__syncthreads();
}




#endif /* BARABASI_GAME_HPP_ */
