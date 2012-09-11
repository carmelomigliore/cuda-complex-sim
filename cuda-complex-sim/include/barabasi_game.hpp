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

	for(int j=0; j<(initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2); j+=2)
	{
		printf("\nLink: %d - %d", links_linearized_array[j], links_linearized_array[j+1]);
	}
}



#endif /* BARABASI_GAME_HPP_ */
