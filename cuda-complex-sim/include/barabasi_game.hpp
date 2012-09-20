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
#include "link.hpp"
#include "parameters.hpp"
#include "curand_kernel.h"
#include "message.hpp"
#include "node.hpp"
#include "templates.hpp"

/* Generates a scale-free network using Barabasi's algorithm */

__device__ void barabasi_game(uint16_t initial_nodes, uint16_t links_number, uint32_t max_nodes, curandState *state)
{
	/* First we allocate links' linearized array, it will contain the target of every link
	 * It will be used to simulate probability.
	 * (initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)
	 */



	uint32_t counter=0; //total link counter

	printf("\nAllocated %d bytes scale-free", (initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)*sizeof(uint32_t));
	if(links_linearized_array==NULL) printf("\nCribba");
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
				//printf("\nDino %1.10f", curand_uniform(state));
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

}

__device__ void  generateMessages(Link* targets_tile, int32_t nodes_number, int32_t this_node, uint16_t ttl)
{
	copyToTile<Link>(links_targets_array, targets_tile,(this_node/(gridDim.x*blockDim.x))*(gridDim.x*blockDim.x), average_links_number, 0); // la divisione restituisce la parte intera del quoziente

	uint32_t random_neighbour_idx;
	uint32_t random_receiver;
	message_t mex;
	random_receiver = (uint32_t)(curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])*nodes_number)%nodes_number;
	mex.original_sender=this_node; mex.receiver=random_receiver; mex.ttl=ttl;
	if(isLinked(random_receiver,targets_tile))
	{
		mex.intermediate=-1;
		//printf("\nInviato vicino %d <-- %d",random_receiver,this_node);
	}
	else
	{
		random_neighbour_idx = (uint32_t)(curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])*average_links_number)%average_links_number;
		mex.intermediate=targets_tile[threadIdx.x*average_links_number+random_neighbour_idx].target;
		//printf("\nInviato random %d ---> ",targets_tile[threadIdx.x*average_links_number+random_neighbour_idx].target);
	}
	//__syncthreads();
	message_array[this_node]=mex;
}

__device__ void checkInbox(Link* targets_tile, int32_t this_thread)
{
	message_t temp = message_array[this_thread];
	uint32_t random_neighbour_idx;
	if(temp.receiver==-1)
	{
		// nothing to do here
	}
	else if(temp.intermediate==-1)
	{
		//printf("\nRicevuto %d da %d al %d° hop",temp.receiver, temp.original_sender, temp.ttl);
		temp.receiver=-1;
	}
	else if(temp.ttl==0)
	{
		temp.receiver=-1;
	}
	else
	{
		memcpy(&targets_tile[threadIdx.x*average_links_number],&links_targets_array[temp.intermediate*average_links_number],average_links_number*sizeof(Link));
		temp.ttl--;
		if(isLinked(temp.receiver,targets_tile))
		{
			temp.intermediate=-1;
		}
		else
		{
			random_neighbour_idx = (uint32_t)(curand_uniform(&cstate[threadIdx.x+blockIdx.x*blockDim.x])*average_links_number)%average_links_number;
			temp.intermediate=targets_tile[threadIdx.x*average_links_number+random_neighbour_idx].target;
		}
	}
	//__syncthreads(); //TODO controlla se è più veloce con o senza syncthread
	message_array[this_thread]=temp;
}


#endif /* BARABASI_GAME_HPP_ */
