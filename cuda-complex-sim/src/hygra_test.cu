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
 * License along with Cuda-complex-sim.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "device.cuh"
#include "host.hpp"
#include "graph_transf.hpp"
#include "h_barabasi_game.hpp"
#include "hygra.cuh"
#include "attributes.hpp"


int main(int argc, char** argv)
{

		bool* nodes_dev;
		Link* links_target_dev;
		bool* flagw_array;
		bool* flagr_array;
		uint32_t* counter;
		task_t* task_dev;
		task_arguments* task_args_dev;
		message_t* inbox_dev;
		n_attribute *prog;
		curandState *d_state;
		coord* attr_array;

//		if (argc!=3)
	//		{
//				perror("\nErrore");
//				exit(1);
//			}


//	uint32_t max_nodes=atoi(argv[1]);
//	uint8_t average_links=atoi(argv[2]);
	uint16_t supplementary_size= 10;




	uint32_t max_nodes = 50000;
	uint8_t average_links= 5;
	uint16_t barabasi_initial_nodes=average_links+1;

	allocateDataStructures(&prog,&nodes_dev, &task_dev, &task_args_dev, &links_target_dev, &inbox_dev,max_nodes,average_links,supplementary_size,&d_state,&flagw_array,&flagr_array,&counter);
	h_allocateDataStructures(supplementary_size,max_nodes,average_links);

	Graph g = h_barabasi_game(barabasi_initial_nodes, 4, max_nodes);
	generatesCoordinates(attr_array);
	copyToDevice(attr_array,(coord*)h_nodes_userattr_array,0,max_nodes);
	startSimulation(links_target_dev,nodes_dev,supplementary_size,g);

	hygra<<<BLOCKS,THREADS_PER_BLOCK,h_average_links_number*THREADS_PER_BLOCK*sizeof(Link)>>>(6,attr_array);

	cudaThreadExit();



  return 0;
}
