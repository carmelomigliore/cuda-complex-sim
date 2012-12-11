/* Copyright (C) 2012  Fabrizio Gueli
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

#ifndef H_BARABASI_GAME_HPP_
#define H_BARABASI_GAME_HPP_

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include "graph_transf.hpp"
#include "h_parameters.hpp"
#include "h_templates.hpp"
#include "h_link.hpp"

/* Generates a scale-free network using Barabasi's algorithm */

__host__ Graph h_barabasi_game(uint16_t initial_nodes, uint16_t links_number, uint32_t max_nodes)
{
	/* First we allocate links' linearized array, it will contain the target of every link
		 * It will be used to simulate probability.
		 * (initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)
		 */

	h_links_linearized_array = (uint32_t*)malloc((initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)*sizeof(uint32_t));
	if(h_links_linearized_array==NULL){
			cerr << "\nCouldn't allocate memory on host 6";
					return false;
		}

	boost::mt19937 gen;
	unsigned int rseed = static_cast<unsigned int>(time(0));
	gen.seed(static_cast<unsigned int>(rseed));


		Graph g;
		uint32_t counter=0; //total link counter

		printf("\nAllocated %d bytes scale-free", (initial_nodes*(initial_nodes-1)*2+(max_nodes-initial_nodes)*links_number*2)*sizeof(uint32_t));

		/* We create the first N nodes (==initial_nodes) and link all of them with one another */

		uint32_t i=0;
		while(i<initial_nodes)
		{
			add_vertex(g);
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
					add_edge(i, j, g);
					h_links_linearized_array[counter]= i; //source and target are added to links_linearized_array
					h_links_linearized_array[counter+1]= j;
					counter+=2;
				}
			}
		}

		/* Now we add one node per time, and add all the links for that node, using Barabasi's algorithm */

		uint32_t random;
		uint32_t random_node;
		bool flag;
		boost::uniform_int<> dist(0, counter);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(gen, dist);
		for(i=initial_nodes; i< max_nodes; i++)
		{
			add_vertex(g);
			for(j=0; j<links_number; j++)
			{
				flag=true;

				/* Let's see to what node the variable random corresponds,
				 * and if it is not already linked, link it. Otherwise, generates a new number.
				 */
				while(flag)
				{
					random = die();		//generates a number between 0 and counter
					random_node=h_links_linearized_array[random];
					if (!(edge(i,random_node,g).second) && random_node!=i)
					{
						flag=false; //exit while
					}
				}
				add_edge(i, random_node, g);
				h_links_linearized_array[counter]= i;			//Add the new link source and target to links_linearized_array
				h_links_linearized_array[counter+1]= random_node;
				counter+=2;
			}
		}
		return g;


}


#endif /* H_BARABASI_GAME_HPP_ */
